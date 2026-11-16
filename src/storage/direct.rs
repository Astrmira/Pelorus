//! 有界共享页缓存、在途缓冲池及 Linux O_DIRECT 预读。

use super::{SsdCacheConfig, SsdIoDriver, SsdIoStats};
use crate::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;

const PAGE: usize = 4096;
const SHARDS: usize = 16;
#[cfg(target_os = "linux")]
mod uring;

#[derive(Default)]
struct PageHasher(u64);

impl std::hash::Hasher for PageHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("PageHasher仅散列u64页号");
    }

    #[inline(always)]
    fn write_u64(&mut self, key: u64) {
        self.0 = (key ^ (key >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    }
}

type PageMap<V> = HashMap<u64, V, std::hash::BuildHasherDefault<PageHasher>>;

struct Lease {
    index: usize,
}

struct PoolState {
    free: Vec<usize>,
    allocated: usize,
    live: usize,
    peak: usize,
}
struct Pool {
    limit: usize,
    slab: memmap2::MmapMut,
    address: std::ptr::NonNull<u8>,
    state: Mutex<PoolState>,
    ready: Condvar,
}
// SAFETY：匿名映射的地址及长度在池生命周期内固定，独占租约由锁唯一分配，发布后的页只读，归还前页持有池的 Arc。
unsafe impl Send for Pool {}
// SAFETY：不同租约覆盖互不重叠的4096字节页，同一租约只在读取完成并发布前写入，最后一个共享读者释放后才能复用。
unsafe impl Sync for Pool {}
impl Pool {
    fn new(limit: usize) -> Result<Self> {
        let length = limit.checked_mul(PAGE).ok_or(Error::CapacityExceeded)?;
        let mut slab = memmap2::MmapMut::map_anon(length)?;
        let address = std::ptr::NonNull::new(slab.as_mut_ptr()).ok_or(Error::CapacityExceeded)?;
        if !(address.as_ptr() as usize).is_multiple_of(PAGE) {
            return Err(Error::InvalidRerankConfig("匿名缓冲池未按4096字节对齐"));
        }
        Ok(Self {
            limit,
            slab,
            address,
            state: Mutex::new(PoolState {
                free: Vec::new(),
                allocated: 0,
                live: 0,
                peak: 0,
            }),
            ready: Condvar::new(),
        })
    }
    fn try_take(&self) -> Option<Lease> {
        let mut state = self.state.lock().unwrap();
        let page = if let Some(page) = state.free.pop() {
            Some(page)
        } else if state.allocated < self.limit {
            let index = state.allocated;
            state.allocated += 1;
            Some(index)
        } else {
            None
        };
        if let Some(page) = page {
            state.live += 1;
            state.peak = state.peak.max(state.live);
            return Some(Lease { index: page });
        }
        None
    }
    fn wait(&self, stopping: &AtomicBool) {
        let state = self.state.lock().unwrap();
        if !stopping.load(Ordering::Acquire)
            && state.free.is_empty()
            && state.allocated >= self.limit
        {
            drop(self.ready.wait(state).unwrap());
        }
    }
    fn put(&self, page: Lease) {
        let mut state = self.state.lock().unwrap();
        state.live -= 1;
        state.free.push(page.index);
        self.ready.notify_one();
    }
    fn write_ptr(&self, lease: &mut Lease) -> *mut u8 {
        // SAFETY：租约由此池唯一分配，索引小于页数，调用方持有独占租约，页尚未发布给共享读者。
        unsafe { self.address.as_ptr().add(lease.index * PAGE) }
    }
}

#[derive(Default)]
struct Counters {
    prefetch_skipped: AtomicU64,
    pending_peak: AtomicU64,
    uring_submitted: AtomicU64,
    uring_completed: AtomicU64,
    uring_max_inflight: AtomicU64,
    uring_submit_batches: AtomicU64,
    uring_completion_batches: AtomicU64,
    uring_completion_batch_max: AtomicU64,
    uring_enter_calls: AtomicU64,
    uring_enter_eintr: AtomicU64,
    ring_errors: AtomicU64,
    registered_buffer_bytes: AtomicU64,
    ring_mapping_bytes: AtomicU64,
    ring_capabilities: AtomicU64,
    read_calls: AtomicU64,
    read_bytes: AtomicU64,
    read_errors: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    pending_hits: AtomicU64,
    query_page_hits: AtomicU64,
    prefetch_requests: AtomicU64,
    prefetch_used: AtomicU64,
    prefetch_wasted: AtomicU64,
    evicted_pages: AtomicU64,
    cached_pages: AtomicU64,
    peak_cached_pages: AtomicU64,
}

struct Page {
    lease: Option<Lease>,
    valid: usize,
    pool: Arc<Pool>,
    counters: Arc<Counters>,
    prefetched: bool,
    consumed: AtomicBool,
}
impl Page {
    fn bytes(&self) -> &[u8] {
        let index = self.lease.as_ref().unwrap().index;
        // SAFETY：读取完成事件已确认，页持有独占编号的租约及池的 Arc，当前共享生命周期内不会复用或修改此页。
        unsafe { std::slice::from_raw_parts(self.pool.address.as_ptr().add(index * PAGE), PAGE) }
    }
    #[inline(always)]
    fn consume(&self) {
        if self.prefetched
            && !self.consumed.load(Ordering::Relaxed)
            && !self.consumed.swap(true, Ordering::Relaxed)
        {
            self.counters.prefetch_used.fetch_add(1, Ordering::Relaxed);
        }
    }
}
impl Drop for Page {
    fn drop(&mut self) {
        if self.prefetched && !self.consumed.load(Ordering::Relaxed) {
            self.counters
                .prefetch_wasted
                .fetch_add(1, Ordering::Relaxed);
        }
        self.pool.put(self.lease.take().unwrap());
    }
}

struct Pending {
    done: AtomicBool,
    result: Mutex<Option<Result<Arc<Page>>>>,
    ready: Condvar,
}
impl Pending {
    fn new() -> Self {
        Self {
            done: AtomicBool::new(false),
            result: Mutex::new(None),
            ready: Condvar::new(),
        }
    }
    /// 无锁检查异步页读取是否已完成（不克隆页句柄）。
    #[inline]
    fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
    /// 非阻塞探测异步页结果而不增加页引用计数。
    #[inline]
    fn check_ready(&self) -> Option<Result<()>> {
        if !self.done.load(Ordering::Acquire) {
            return None;
        }
        Some(match self.result.lock().unwrap().as_ref().unwrap() {
            Ok(_) => Ok(()),
            Err(error) => Err(error.clone()),
        })
    }
    #[inline]
    fn try_ready(&self) -> Option<Result<Arc<Page>>> {
        if !self.done.load(Ordering::Acquire) {
            return None;
        }
        self.result.lock().unwrap().as_ref().cloned()
    }
    fn wait(&self) -> Result<Arc<Page>> {
        let mut state = self.result.lock().unwrap();
        while state.is_none() {
            state = self.ready.wait(state).unwrap();
        }
        state.as_ref().unwrap().clone()
    }
    fn finish(&self, result: Result<Arc<Page>>) {
        *self.result.lock().unwrap() = Some(result);
        self.done.store(true, Ordering::Release);
        self.ready.notify_all();
    }
}

enum Entry {
    Ready(Arc<Page>),
    Loading(Arc<Pending>),
}
struct Cache {
    entries: PageMap<Entry>,
    fifo: VecDeque<u64>,
    resident: usize,
    capacity: usize,
}
struct Task {
    page: u64,
    result: Arc<Pending>,
    prefetched: bool,
}
struct Queue {
    tasks: VecDeque<Task>,
    active: usize,
    shutdown: bool,
    failure: Option<Error>,
}
struct Shared {
    file: File,
    file_len: usize,
    caches: Vec<RwLock<Cache>>,
    evict_cursor: std::sync::atomic::AtomicUsize,
    pending_limit: usize,
    failed: AtomicBool,
    shutting_down: AtomicBool,
    pool: Arc<Pool>,
    counters: Arc<Counters>,
    queue: Mutex<Queue>,
    available: Condvar,
    idle: Condvar,
}

impl Shared {
    fn check_error(&self) -> Result<()> {
        if self.failed.load(Ordering::Acquire) {
            return Err(self.queue.lock().unwrap().failure.as_ref().unwrap().clone());
        }
        Ok(())
    }

    fn cached(&self, page: u64, prefetch: bool) -> Option<Entry> {
        let cache = self.caches[page as usize % SHARDS].read().unwrap();
        cache.entries.get(&page).map(|entry| match entry {
            Entry::Ready(data) => {
                if !prefetch {
                    self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
                }
                Entry::Ready(Arc::clone(data))
            }
            Entry::Loading(pending) => {
                if !prefetch {
                    self.counters.pending_hits.fetch_add(1, Ordering::Relaxed);
                }
                Entry::Loading(Arc::clone(pending))
            }
        })
    }

    /// 仅非阻塞探测页是否已就绪（不计入缓存命中统计，供小缓存非固定批状态推进）。
    fn is_page_ready(&self, page: u64) -> Result<bool> {
        let cache = self.caches[page as usize % SHARDS].read().unwrap();
        match cache.entries.get(&page) {
            Some(Entry::Ready(_)) => Ok(true),
            Some(Entry::Loading(pending)) => match pending.check_ready() {
                Some(Ok(())) => Ok(true),
                Some(Err(error)) => Err(error),
                None => Ok(false),
            },
            None => Ok(false),
        }
    }

    fn request(&self, page: u64, prefetch: bool) -> Result<Option<Entry>> {
        self.check_error()?;
        if prefetch {
            self.counters
                .prefetch_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        if let Some(entry) = self.cached(page, prefetch) {
            return Ok(Some(entry));
        }
        let mut queue = self.queue.lock().unwrap();
        loop {
            if let Some(error) = &queue.failure {
                return Err(error.clone());
            }
            if queue.shutdown {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "SSD读取后端已关闭",
                )
                .into());
            }
            if let Some(entry) = self.cached(page, prefetch) {
                return Ok(Some(entry));
            }
            if queue.tasks.len() + queue.active < self.pending_limit {
                break;
            }
            if prefetch {
                self.counters
                    .prefetch_skipped
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            queue = self.available.wait(queue).unwrap();
        }
        if !prefetch {
            self.counters.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
        let pending = Arc::new(Pending::new());
        self.caches[page as usize % SHARDS]
            .write()
            .unwrap()
            .entries
            .insert(page, Entry::Loading(Arc::clone(&pending)));
        queue.tasks.push_back(Task {
            page,
            result: Arc::clone(&pending),
            prefetched: prefetch,
        });
        self.counters
            .pending_peak
            .fetch_max((queue.tasks.len() + queue.active) as u64, Ordering::Relaxed);
        self.available.notify_one();
        Ok(Some(Entry::Loading(pending)))
    }

    fn try_buffer(&self) -> Option<Lease> {
        loop {
            if let Some(lease) = self.pool.try_take() {
                return Some(lease);
            }
            let start = self.evict_cursor.fetch_add(1, Ordering::Relaxed);
            let mut evicted = None;
            for offset in 0..SHARDS {
                let mut cache = self.caches[(start + offset) % SHARDS].write().unwrap();
                if let Some(victim) = cache.fifo.pop_front() {
                    evicted = cache.entries.remove(&victim);
                    cache.resident -= 1;
                    self.counters.cached_pages.fetch_sub(1, Ordering::Relaxed);
                    self.counters.evicted_pages.fetch_add(1, Ordering::Relaxed);
                    break;
                }
            }
            drop(evicted?);
        }
    }

    fn buffer(&self) -> Result<Lease> {
        loop {
            if let Some(lease) = self.try_buffer() {
                return Ok(lease);
            }
            if self.shutting_down.load(Ordering::Acquire) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "SSD读取后端已关闭",
                )
                .into());
            }
            self.pool.wait(&self.shutting_down);
        }
    }

    fn read(&self, page: u64, prefetched: bool) -> Result<Arc<Page>> {
        let offset = usize::try_from(page)
            .ok()
            .and_then(|value| value.checked_mul(PAGE))
            .ok_or(Error::CapacityExceeded)?;
        if offset >= self.file_len {
            return Err(Error::CorruptIndex("SSD 读取页超出文件"));
        }
        let mut data = Page {
            lease: Some(self.buffer()?),
            valid: 0,
            pool: Arc::clone(&self.pool),
            counters: Arc::clone(&self.counters),
            prefetched,
            consumed: AtomicBool::new(false),
        };
        let expected = PAGE.min(self.file_len - offset);
        loop {
            self.counters.read_calls.fetch_add(1, Ordering::Relaxed);
            let pointer = self.pool.write_ptr(data.lease.as_mut().unwrap());
            // SAFETY：当前页未发布，唯一租约覆盖4096个可写字节，同步 read_at 返回后才会交给 Arc 读者。
            let bytes = unsafe { std::slice::from_raw_parts_mut(pointer, PAGE) };
            let result = read_at(&self.file, bytes, offset as u64);
            match result {
                Ok(count) => {
                    self.counters
                        .read_bytes
                        .fetch_add(count as u64, Ordering::Relaxed);
                    if count != expected {
                        self.counters.read_errors.fetch_add(1, Ordering::Relaxed);
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "SSD 页短读",
                        )
                        .into());
                    }
                    data.valid = count;
                    return Ok(Arc::new(data));
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    self.counters.read_errors.fetch_add(1, Ordering::Relaxed);
                    return Err(error.into());
                }
            }
        }
    }

    fn worker(&self) {
        loop {
            let task = {
                let mut queue = self.queue.lock().unwrap();
                while queue.tasks.is_empty() && !queue.shutdown {
                    queue = self.available.wait(queue).unwrap();
                }
                if queue.tasks.is_empty() && queue.shutdown {
                    return;
                }
                queue.active += 1;
                queue.tasks.pop_front().unwrap()
            };
            let result = self.read(task.page, task.prefetched);
            self.complete(task, result);
        }
    }

    fn complete(&self, task: Task, result: Result<Arc<Page>>) {
        {
            let mut cache = self.caches[task.page as usize % SHARDS].write().unwrap();
            cache.entries.remove(&task.page);
            if let Ok(page) = &result
                && cache.capacity > 0
            {
                while cache.resident >= cache.capacity {
                    let victim = cache.fifo.pop_front().unwrap();
                    cache.entries.remove(&victim);
                    cache.resident -= 1;
                    self.counters.cached_pages.fetch_sub(1, Ordering::Relaxed);
                    self.counters.evicted_pages.fetch_add(1, Ordering::Relaxed);
                }
                cache
                    .entries
                    .insert(task.page, Entry::Ready(Arc::clone(page)));
                cache.fifo.push_back(task.page);
                cache.resident += 1;
                let current = self.counters.cached_pages.fetch_add(1, Ordering::Relaxed) + 1;
                self.counters
                    .peak_cached_pages
                    .fetch_max(current, Ordering::Relaxed);
            }
        }
        task.result.finish(result);
        drop(task);
        self.finish_io();
    }

    fn finish_io(&self) {
        let mut queue = self.queue.lock().unwrap();
        queue.active -= 1;
        self.available.notify_all();
        if queue.active == 0 && queue.tasks.is_empty() {
            self.idle.notify_all();
        }
    }

    fn reject(&self, task: &Task, error: Error) {
        self.caches[task.page as usize % SHARDS]
            .write()
            .unwrap()
            .entries
            .remove(&task.page);
        task.result.finish(Err(error));
    }

    fn fail_queued(&self, error: Error) {
        let tasks = {
            let mut queue = self.queue.lock().unwrap();
            queue.failure.get_or_insert_with(|| error.clone());
            self.failed.store(true, Ordering::Release);
            queue.tasks.drain(..).collect::<Vec<_>>()
        };
        for task in tasks {
            self.reject(&task, error.clone());
        }
        self.available.notify_all();
        self.idle.notify_all();
    }
}

pub(crate) struct DirectReader {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    pub(crate) config: SsdCacheConfig,
    offset: usize,
    pub(crate) length: usize,
    /// Stage 2 自校准认证门控阈值（位模式存放 f64，0.0 表示关闭并维持 100% CRC 无损全重排）。
    selective_gate_bits: AtomicU64,
}

enum BatchPageState {
    Ready(Arc<Page>, bool),
    Loading(Arc<Pending>),
    Cached(u64),
    Queued(u64),
    Demand(u64),
}

#[derive(Clone, Copy)]
enum SlotMapping {
    Fallback,
    Single {
        batch_idx: usize,
        lo: usize,
        hi: usize,
    },
    Multi {
        first_ref: usize,
        page_count: usize,
        lo: usize,
        end_hi: usize,
    },
}

pub(crate) struct DirectScratch {
    pub(crate) bytes: Vec<u8>,
    last: Option<(u64, Arc<Page>)>,
    batch_pages: Vec<(u64, BatchPageState)>,
    slot_mappings: Vec<SlotMapping>,
    multi_page_indices: Vec<usize>,
    tmp_pages: Vec<u64>,
    shard_buckets: [Vec<usize>; SHARDS],
    shard_misses: [Vec<usize>; SHARDS],
    cached_hits_acc: u64,
    pending_hits_acc: u64,
    query_page_hits_acc: u64,
    last_page_idx: Option<usize>,
}

impl DirectScratch {
    /// 清空单次查询残留页句柄与状态标记，保留各 Vec 容量以实现稳态零堆分配。
    #[inline]
    pub(crate) fn reset_query(&mut self) {
        self.last = None;
        self.last_page_idx = None;
        self.batch_pages.clear();
        self.slot_mappings.clear();
        self.multi_page_indices.clear();
        self.cached_hits_acc = 0;
        self.pending_hits_acc = 0;
        self.query_page_hits_acc = 0;
    }

    /// 在阻塞等待 I/O 前释放批内已固定页句柄与待决引用，防止小缓存并发持满全部租约。
    #[inline]
    fn release_pinned_pages(&mut self) {
        self.last = None;
        self.last_page_idx = None;
        for (page, state) in &mut self.batch_pages {
            if matches!(
                state,
                BatchPageState::Ready(_, _) | BatchPageState::Loading(_)
            ) {
                *state = BatchPageState::Demand(*page);
            }
        }
    }
}

impl Default for DirectScratch {
    fn default() -> Self {
        Self {
            bytes: Vec::new(),
            last: None,
            batch_pages: Vec::new(),
            slot_mappings: Vec::new(),
            multi_page_indices: Vec::new(),
            tmp_pages: Vec::new(),
            shard_buckets: std::array::from_fn(|_| Vec::new()),
            shard_misses: std::array::from_fn(|_| Vec::new()),
            cached_hits_acc: 0,
            pending_hits_acc: 0,
            query_page_hits_acc: 0,
            last_page_idx: None,
        }
    }
}

impl DirectReader {
    pub(crate) fn open(
        source: &File,
        offset: usize,
        length: usize,
        config: SsdCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        let file = open_direct(source)?;
        Self::from_file(file, offset, length, config)
    }
    fn from_file(file: File, offset: usize, length: usize, config: SsdCacheConfig) -> Result<Self> {
        config.validate()?;
        let file_len =
            usize::try_from(file.metadata()?.len()).map_err(|_| Error::CapacityExceeded)?;
        if offset.checked_add(length).is_none_or(|end| end > file_len) {
            return Err(Error::CorruptIndex("SSD 向量区间越界"));
        }
        let pages = config.cache_bytes / PAGE;
        let caches = (0..SHARDS)
            .map(|shard| {
                RwLock::new(Cache {
                    entries: PageMap::default(),
                    fifo: VecDeque::new(),
                    resident: 0,
                    capacity: pages / SHARDS + usize::from(shard < pages % SHARDS),
                })
            })
            .collect();
        let shared = Arc::new(Shared {
            file,
            file_len,
            caches,
            evict_cursor: std::sync::atomic::AtomicUsize::new(0),
            pending_limit: config.inflight_bytes / PAGE,
            failed: AtomicBool::new(false),
            shutting_down: AtomicBool::new(false),
            pool: Arc::new(Pool::new(
                (config.cache_bytes + config.inflight_bytes) / PAGE,
            )?),
            counters: Arc::new(Counters::default()),
            queue: Mutex::new(Queue {
                tasks: VecDeque::new(),
                active: 0,
                shutdown: false,
                failure: None,
            }),
            available: Condvar::new(),
            idle: Condvar::new(),
        });
        let mut reader = Self {
            shared,
            workers: Vec::new(),
            config,
            offset,
            length,
            selective_gate_bits: AtomicU64::new(0.0f64.to_bits()),
        };
        if config.driver == SsdIoDriver::IoUring {
            #[cfg(target_os = "linux")]
            {
                let (sender, receiver) = std::sync::mpsc::sync_channel(1);
                let shared = Arc::clone(&reader.shared);
                reader.workers.push(
                    std::thread::Builder::new()
                        .name("SSD异步调度".into())
                        .spawn(move || {
                            let ring = match uring::Ring::new(&shared) {
                                Ok(ring) => ring,
                                Err(error) => {
                                    let _ = sender.send(Err(error));
                                    return;
                                }
                            };
                            if sender.send(Ok(())).is_ok() {
                                shared.uring_worker(ring);
                            }
                        })?,
                );
                receiver
                    .recv()
                    .map_err(|_| std::io::Error::other("io_uring初始化线程未完成握手"))??;
                return Ok(reader);
            }
            #[cfg(not(target_os = "linux"))]
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "io_uring需要Linux内核",
            )
            .into());
        }
        for number in 0..config.io_workers {
            let shared = Arc::clone(&reader.shared);
            reader.workers.push(
                std::thread::Builder::new()
                    .name(format!("SSD读取{number}"))
                    .spawn(move || shared.worker())?,
            );
        }
        Ok(reader)
    }

    fn range(&self, slot: usize, stride: usize) -> Result<(usize, usize)> {
        let relative = slot.checked_mul(stride).ok_or(Error::CapacityExceeded)?;
        let end = relative
            .checked_add(stride)
            .ok_or(Error::CapacityExceeded)?;
        if end > self.length {
            return Err(Error::CorruptIndex("SSD 重排记录越界"));
        }
        Ok((self.offset + relative, self.offset + end))
    }

    pub(crate) fn prefetch(&self, slot: usize, stride: usize) {
        let Ok((start, end)) = self.range(slot, stride) else {
            return;
        };
        for page in start / PAGE..=(end - 1) / PAGE {
            let _ = self.shared.request(page as u64, true);
        }
    }

    /// 配置 Stage 2 自校准选择性认证门控的高斯置信倍數（`None` 表示关闭并保持全候选无损精排）。
    #[inline]
    pub(crate) fn set_selective_gating(&self, sigma: Option<f64>) {
        let val = sigma.unwrap_or(0.0);
        self.selective_gate_bits
            .store(val.to_bits(), Ordering::Relaxed);
    }

    /// 读取当前生效的 Stage 2 认证门控倍数；关闭时返回 `None`。
    #[inline]
    pub(crate) fn selective_gating(&self) -> Option<f64> {
        let v = f64::from_bits(self.selective_gate_bits.load(Ordering::Relaxed));
        (v > 0.0 && v.is_finite()).then_some(v)
    }

    /// 当缓冲池容量与在途预算充足时才在批内直接固定页句柄；小缓存只记录页号以防并发占满租约。
    #[inline]
    fn pin_cached(&self) -> bool {
        self.shared.pending_limit >= 2 && self.shared.pool.limit >= 64
    }

    /// 将缓存条目转换为批页状态：大缓存固定句柄，小缓存仅记录页号。
    #[inline]
    fn batch_state_from_entry(page: u64, entry: &Entry, pin_cached: bool) -> BatchPageState {
        match entry {
            Entry::Ready(data) => {
                if pin_cached {
                    BatchPageState::Ready(Arc::clone(data), false)
                } else {
                    BatchPageState::Cached(page)
                }
            }
            Entry::Loading(pending) => {
                if pin_cached {
                    BatchPageState::Loading(Arc::clone(pending))
                } else {
                    BatchPageState::Queued(page)
                }
            }
        }
    }

    /// 批量预解析并一次加锁入队全部缺失页，消除滑动小波次带来的碎片化 io_uring 提交与重复锁查找。
    pub(crate) fn prepare_rerank_batch(
        &self,
        slots: &[usize],
        stride: usize,
        scratch: &mut DirectScratch,
    ) {
        self.prepare_rerank_batch_impl(slots, stride, scratch, true);
    }

    /// Stage 2 仅预映射槽位并拉取当前已命中内存页，未命中页留作 Demand 供门控后按需入队。
    pub(crate) fn probe_rerank_batch_hits(
        &self,
        slots: &[usize],
        stride: usize,
        scratch: &mut DirectScratch,
    ) {
        self.prepare_rerank_batch_impl(slots, stride, scratch, false);
    }

    fn prepare_rerank_batch_impl(
        &self,
        slots: &[usize],
        stride: usize,
        scratch: &mut DirectScratch,
        enqueue_all: bool,
    ) {
        scratch.batch_pages.clear();
        scratch.slot_mappings.clear();
        scratch.multi_page_indices.clear();
        scratch.tmp_pages.clear();
        for bucket in &mut scratch.shard_buckets {
            bucket.clear();
        }
        for misses in &mut scratch.shard_misses {
            misses.clear();
        }
        scratch.last_page_idx = None;
        scratch.cached_hits_acc = 0;
        scratch.pending_hits_acc = 0;
        scratch.query_page_hits_acc = 0;

        if self.config.prefetch_window == 0
            || self.config.cache_bytes == 0
            || stride == 0
            || slots.is_empty()
        {
            return;
        }
        scratch.last = None;
        let cache_pages = self.config.cache_bytes / PAGE;
        let max_pages = (cache_pages / 2)
            .min(self.shared.pending_limit)
            .clamp(1, 1024);

        for &slot in slots {
            let Ok((start, end)) = self.range(slot, stride) else {
                continue;
            };
            for page in start / PAGE..=(end - 1) / PAGE {
                scratch.tmp_pages.push(page as u64);
            }
        }
        if scratch.tmp_pages.is_empty() {
            scratch
                .slot_mappings
                .resize(slots.len(), SlotMapping::Fallback);
            return;
        }
        scratch.tmp_pages.sort_unstable();
        scratch.tmp_pages.dedup();
        if scratch.tmp_pages.len() > max_pages {
            scratch.tmp_pages.truncate(max_pages);
        }

        scratch.batch_pages.reserve(scratch.tmp_pages.len());
        for (idx, &page) in scratch.tmp_pages.iter().enumerate() {
            scratch
                .batch_pages
                .push((page, BatchPageState::Demand(page)));
            scratch.shard_buckets[page as usize % SHARDS].push(idx);
        }

        scratch.slot_mappings.reserve(slots.len());
        let mut prefetched_touches = 0u64;
        for &slot in slots {
            let Ok((start, end)) = self.range(slot, stride) else {
                scratch.slot_mappings.push(SlotMapping::Fallback);
                continue;
            };
            let p0 = (start / PAGE) as u64;
            let p1 = ((end - 1) / PAGE) as u64;
            if p0 == p1 {
                if let Ok(batch_idx) = scratch.tmp_pages.binary_search(&p0) {
                    let lo = start - (p0 as usize) * PAGE;
                    scratch.slot_mappings.push(SlotMapping::Single {
                        batch_idx,
                        lo,
                        hi: lo + stride,
                    });
                    prefetched_touches += 1;
                } else {
                    scratch.slot_mappings.push(SlotMapping::Fallback);
                }
            } else {
                let first_ref = scratch.multi_page_indices.len();
                let mut all_ok = true;
                for p in p0..=p1 {
                    if let Ok(batch_idx) = scratch.tmp_pages.binary_search(&p) {
                        scratch.multi_page_indices.push(batch_idx);
                    } else {
                        all_ok = false;
                        break;
                    }
                }
                if all_ok {
                    let page_count = scratch.multi_page_indices.len() - first_ref;
                    let lo = start - (p0 as usize) * PAGE;
                    let end_hi = end - (p1 as usize) * PAGE;
                    scratch.slot_mappings.push(SlotMapping::Multi {
                        first_ref,
                        page_count,
                        lo,
                        end_hi,
                    });
                    prefetched_touches += page_count as u64;
                } else {
                    scratch.multi_page_indices.truncate(first_ref);
                    scratch.slot_mappings.push(SlotMapping::Fallback);
                }
            }
        }
        if enqueue_all && prefetched_touches > 0 {
            self.shared
                .counters
                .prefetch_requests
                .fetch_add(prefetched_touches, Ordering::Relaxed);
        }

        let pin = self.pin_cached();
        let mut has_misses = false;
        for shard in 0..SHARDS {
            let bucket = &scratch.shard_buckets[shard];
            if bucket.is_empty() {
                continue;
            }
            let guard = self.shared.caches[shard].read().unwrap();
            for &u_idx in bucket {
                let page = scratch.batch_pages[u_idx].0;
                match guard.entries.get(&page) {
                    Some(entry) => {
                        scratch.batch_pages[u_idx].1 =
                            Self::batch_state_from_entry(page, entry, pin);
                    }
                    None => {
                        if enqueue_all {
                            scratch.shard_misses[shard].push(u_idx);
                            has_misses = true;
                        }
                    }
                }
            }
        }
        if enqueue_all && has_misses {
            self.enqueue_shard_misses(scratch);
        }
    }

    /// Stage 2 仅对选定位置集合按分片归集仍处于 Demand 的物理页并单波批量提交 io_uring。
    pub(crate) fn enqueue_rerank_positions(
        &self,
        positions: &[usize],
        scratch: &mut DirectScratch,
    ) {
        if positions.is_empty() || scratch.slot_mappings.is_empty() {
            return;
        }
        for misses in &mut scratch.shard_misses {
            misses.clear();
        }
        let mut touches = 0u64;
        for &pos in positions {
            let Some(&mapping) = scratch.slot_mappings.get(pos) else {
                continue;
            };
            match mapping {
                SlotMapping::Fallback => {}
                SlotMapping::Single { batch_idx, .. } => {
                    touches += 1;
                    if let BatchPageState::Demand(page) = scratch.batch_pages[batch_idx].1 {
                        scratch.shard_misses[page as usize % SHARDS].push(batch_idx);
                    }
                }
                SlotMapping::Multi {
                    first_ref,
                    page_count,
                    ..
                } => {
                    touches += page_count as u64;
                    for k in 0..page_count {
                        let batch_idx = scratch.multi_page_indices[first_ref + k];
                        if let BatchPageState::Demand(page) = scratch.batch_pages[batch_idx].1 {
                            scratch.shard_misses[page as usize % SHARDS].push(batch_idx);
                        }
                    }
                }
            }
        }
        if touches > 0 {
            self.shared
                .counters
                .prefetch_requests
                .fetch_add(touches, Ordering::Relaxed);
        }
        let mut has_misses = false;
        for misses in &mut scratch.shard_misses {
            if !misses.is_empty() {
                misses.sort_unstable();
                misses.dedup();
                has_misses = true;
            }
        }
        if has_misses {
            self.enqueue_shard_misses(scratch);
        }
    }

    /// 在单一 `queue` 锁内依次检查写分片缓存并发布缺失页任务，最后唤醒一次 I/O 工作线程。
    fn enqueue_shard_misses(&self, scratch: &mut DirectScratch) {
        let pin = self.pin_cached();
        let mut pushed = 0u64;
        let mut skipped = 0u64;
        {
            let mut queue = self.shared.queue.lock().unwrap();
            if queue.failure.is_none() && !queue.shutdown {
                let mut avail = self
                    .shared
                    .pending_limit
                    .saturating_sub(queue.tasks.len() + queue.active);
                for shard in 0..SHARDS {
                    let misses = &scratch.shard_misses[shard];
                    if misses.is_empty() {
                        continue;
                    }
                    let mut guard = self.shared.caches[shard].write().unwrap();
                    for &u_idx in misses {
                        let page = scratch.batch_pages[u_idx].0;
                        match guard.entries.get(&page) {
                            Some(entry) => {
                                scratch.batch_pages[u_idx].1 =
                                    Self::batch_state_from_entry(page, entry, pin);
                            }
                            None => {
                                if avail > 0 {
                                    avail -= 1;
                                    pushed += 1;
                                    let pending = Arc::new(Pending::new());
                                    guard
                                        .entries
                                        .insert(page, Entry::Loading(Arc::clone(&pending)));
                                    queue.tasks.push_back(Task {
                                        page,
                                        result: Arc::clone(&pending),
                                        prefetched: true,
                                    });
                                    scratch.batch_pages[u_idx].1 = if pin {
                                        BatchPageState::Loading(pending)
                                    } else {
                                        BatchPageState::Queued(page)
                                    };
                                } else {
                                    skipped += 1;
                                }
                            }
                        }
                    }
                }
                if pushed > 0 {
                    self.shared
                        .counters
                        .pending_peak
                        .fetch_max((queue.tasks.len() + queue.active) as u64, Ordering::Relaxed);
                }
            }
        }
        if skipped > 0 {
            self.shared
                .counters
                .prefetch_skipped
                .fetch_add(skipped, Ordering::Relaxed);
        }
        if pushed > 0 {
            self.shared.available.notify_one();
        }
    }

    #[inline]
    pub(crate) fn is_slot_ready(&self, pos: usize, scratch: &DirectScratch) -> bool {
        let Some(&mapping) = scratch.slot_mappings.get(pos) else {
            return false;
        };
        match mapping {
            SlotMapping::Fallback => false,
            SlotMapping::Single { batch_idx, .. } => {
                matches!(
                    scratch.batch_pages[batch_idx].1,
                    BatchPageState::Ready(_, _) | BatchPageState::Cached(_)
                )
            }
            SlotMapping::Multi {
                first_ref,
                page_count,
                ..
            } => (0..page_count).all(|k| {
                let idx = scratch.multi_page_indices[first_ref + k];
                matches!(
                    scratch.batch_pages[idx].1,
                    BatchPageState::Ready(_, _) | BatchPageState::Cached(_)
                )
            }),
        }
    }

    #[inline]
    pub(crate) fn try_upgrade_slot_ready(
        &self,
        pos: usize,
        scratch: &mut DirectScratch,
    ) -> Result<bool> {
        let Some(&mapping) = scratch.slot_mappings.get(pos) else {
            return Ok(false);
        };
        match mapping {
            SlotMapping::Fallback => Ok(false),
            SlotMapping::Single { batch_idx, .. } => self.try_upgrade_page(batch_idx, scratch),
            SlotMapping::Multi {
                first_ref,
                page_count,
                ..
            } => {
                // 跨页条目仅在关联物理页全部无需等待时整体升级，避免部分页提前固定租约。
                for k in 0..page_count {
                    let idx = scratch.multi_page_indices[first_ref + k];
                    if !self.is_batch_page_ready(idx, scratch)? {
                        return Ok(false);
                    }
                }
                for k in 0..page_count {
                    let idx = scratch.multi_page_indices[first_ref + k];
                    if !self.try_upgrade_page(idx, scratch)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    #[inline]
    fn is_batch_page_ready(&self, batch_idx: usize, scratch: &DirectScratch) -> Result<bool> {
        match &scratch.batch_pages[batch_idx].1 {
            BatchPageState::Ready(_, _) | BatchPageState::Cached(_) => Ok(true),
            BatchPageState::Loading(pending) => match pending.check_ready() {
                Some(Ok(())) => Ok(true),
                Some(Err(error)) => Err(error),
                None => Ok(false),
            },
            BatchPageState::Queued(page) => self.shared.is_page_ready(*page),
            BatchPageState::Demand(_) => Ok(false),
        }
    }

    #[inline]
    fn try_upgrade_page(&self, batch_idx: usize, scratch: &mut DirectScratch) -> Result<bool> {
        match &scratch.batch_pages[batch_idx].1 {
            BatchPageState::Ready(_, _) | BatchPageState::Cached(_) => Ok(true),
            BatchPageState::Loading(pending) => match pending.try_ready() {
                Some(Ok(page)) => {
                    scratch.batch_pages[batch_idx].1 = BatchPageState::Ready(page, true);
                    Ok(true)
                }
                Some(Err(error)) => Err(error),
                None => Ok(false),
            },
            BatchPageState::Queued(page) => {
                let page = *page;
                if self.shared.is_page_ready(page)? {
                    scratch.batch_pages[batch_idx].1 = BatchPageState::Cached(page);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            BatchPageState::Demand(_) => Ok(false),
        }
    }

    #[inline]
    fn account_ready_hit(batch_idx: usize, scratch: &mut DirectScratch) {
        if let BatchPageState::Ready(_, from_loading) = &mut scratch.batch_pages[batch_idx].1 {
            if scratch.last_page_idx == Some(batch_idx) {
                scratch.query_page_hits_acc += 1;
            } else if *from_loading {
                *from_loading = false;
                scratch.pending_hits_acc += 1;
                scratch.last_page_idx = Some(batch_idx);
            } else {
                scratch.cached_hits_acc += 1;
                scratch.last_page_idx = Some(batch_idx);
            }
        }
    }

    #[inline]
    fn resolve_batch_page_ready(batch_idx: usize, scratch: &mut DirectScratch) -> Result<()> {
        match &mut scratch.batch_pages[batch_idx].1 {
            BatchPageState::Ready(_, _) => {
                Self::account_ready_hit(batch_idx, scratch);
                Ok(())
            }
            BatchPageState::Loading(pending) => {
                let page = pending.wait()?;
                scratch.batch_pages[batch_idx].1 = BatchPageState::Ready(page, false);
                scratch.pending_hits_acc += 1;
                scratch.last_page_idx = Some(batch_idx);
                Ok(())
            }
            BatchPageState::Cached(_) | BatchPageState::Queued(_) | BatchPageState::Demand(_) => {
                unreachable!()
            }
        }
    }

    #[inline]
    fn store_single_last_page(
        batch_idx: usize,
        page_num: u64,
        lo: usize,
        hi: usize,
        page: Arc<Page>,
        scratch: &mut DirectScratch,
    ) -> Result<&[u8]> {
        page.consume();
        if hi > page.valid {
            return Err(Error::CorruptIndex("SSD 页不包含完整记录"));
        }
        scratch.batch_pages[batch_idx].1 = BatchPageState::Cached(page_num);
        scratch.last = Some((page_num, page));
        Ok(&scratch.last.as_ref().unwrap().1.bytes()[lo..hi])
    }

    pub(crate) fn record_prepared<'a>(
        &self,
        pos: usize,
        slot: usize,
        stride: usize,
        scratch: &'a mut DirectScratch,
    ) -> Result<&'a [u8]> {
        let Some(&mapping) = scratch.slot_mappings.get(pos) else {
            return self.record(slot, stride, scratch);
        };
        match mapping {
            SlotMapping::Fallback => self.record(slot, stride, scratch),
            SlotMapping::Single { batch_idx, lo, hi } => {
                let page_num = scratch.batch_pages[batch_idx].0;
                match &scratch.batch_pages[batch_idx].1 {
                    BatchPageState::Demand(_) => self.record(slot, stride, scratch),
                    BatchPageState::Ready(_, _) => {
                        self.shared.check_error()?;
                        Self::account_ready_hit(batch_idx, scratch);
                        let BatchPageState::Ready(ref page, _) = scratch.batch_pages[batch_idx].1
                        else {
                            unreachable!();
                        };
                        page.consume();
                        if hi > page.valid {
                            return Err(Error::CorruptIndex("SSD 页不包含完整记录"));
                        }
                        Ok(&page.bytes()[lo..hi])
                    }
                    BatchPageState::Loading(_) => {
                        self.shared.check_error()?;
                        let ready = {
                            let BatchPageState::Loading(ref pending) =
                                scratch.batch_pages[batch_idx].1
                            else {
                                unreachable!();
                            };
                            pending.try_ready()
                        };
                        let page = match ready {
                            Some(result) => result?,
                            None => {
                                let BatchPageState::Loading(pending) = std::mem::replace(
                                    &mut scratch.batch_pages[batch_idx].1,
                                    BatchPageState::Demand(page_num),
                                ) else {
                                    unreachable!();
                                };
                                // 必须先归还批内已固定的所有页租约再阻塞等待，避免小缓存死锁。
                                scratch.release_pinned_pages();
                                pending.wait()?
                            }
                        };
                        scratch.pending_hits_acc += 1;
                        if self.pin_cached() {
                            page.consume();
                            if hi > page.valid {
                                return Err(Error::CorruptIndex("SSD 页不包含完整记录"));
                            }
                            scratch.batch_pages[batch_idx].1 = BatchPageState::Ready(page, false);
                            scratch.last_page_idx = Some(batch_idx);
                            let BatchPageState::Ready(ref pinned, _) =
                                scratch.batch_pages[batch_idx].1
                            else {
                                unreachable!();
                            };
                            Ok(&pinned.bytes()[lo..hi])
                        } else {
                            Self::store_single_last_page(batch_idx, page_num, lo, hi, page, scratch)
                        }
                    }
                    BatchPageState::Cached(_) | BatchPageState::Queued(_) => {
                        self.shared.check_error()?;
                        if let Some((last, data)) = &scratch.last
                            && *last == page_num
                        {
                            scratch.query_page_hits_acc += 1;
                            data.consume();
                            if hi > data.valid {
                                return Err(Error::CorruptIndex("SSD 页不包含完整记录"));
                            }
                            return Ok(&scratch.last.as_ref().unwrap().1.bytes()[lo..hi]);
                        }
                        scratch.last = None;
                        let Some(entry) = self.shared.cached(page_num, false) else {
                            return self.record(slot, stride, scratch);
                        };
                        let data = match entry {
                            Entry::Ready(data) => data,
                            Entry::Loading(pending) => {
                                if let Some(result) = pending.try_ready() {
                                    result?
                                } else {
                                    scratch.release_pinned_pages();
                                    pending.wait()?
                                }
                            }
                        };
                        Self::store_single_last_page(batch_idx, page_num, lo, hi, data, scratch)
                    }
                }
            }
            SlotMapping::Multi {
                first_ref,
                page_count,
                lo,
                end_hi,
            } => {
                // 跨页记录仅当关联物理页全部无需等待时走零等待合并，否则释放固定页走逐页安全读取。
                let all_nonblocking = (0..page_count).all(|k| {
                    let idx = scratch.multi_page_indices[first_ref + k];
                    match &scratch.batch_pages[idx].1 {
                        BatchPageState::Ready(_, _) => true,
                        BatchPageState::Loading(pending) => pending.is_done(),
                        BatchPageState::Cached(_)
                        | BatchPageState::Queued(_)
                        | BatchPageState::Demand(_) => false,
                    }
                });
                if !all_nonblocking {
                    return self.record(slot, stride, scratch);
                }
                self.shared.check_error()?;
                scratch.bytes.clear();
                scratch.bytes.reserve(stride);
                for k in 0..page_count {
                    let idx = scratch.multi_page_indices[first_ref + k];
                    Self::resolve_batch_page_ready(idx, scratch)?;
                    let BatchPageState::Ready(ref page, _) = scratch.batch_pages[idx].1 else {
                        unreachable!();
                    };
                    page.consume();
                    let p_lo = if k == 0 { lo } else { 0 };
                    let p_hi = if k + 1 == page_count { end_hi } else { PAGE };
                    if p_hi > page.valid {
                        return Err(Error::CorruptIndex("SSD 页不包含完整记录"));
                    }
                    scratch.bytes.extend_from_slice(&page.bytes()[p_lo..p_hi]);
                }
                Ok(&scratch.bytes)
            }
        }
    }

    pub(crate) fn finish_rerank_batch(&self, scratch: &mut DirectScratch) {
        if scratch.cached_hits_acc > 0 {
            self.shared
                .counters
                .cache_hits
                .fetch_add(scratch.cached_hits_acc, Ordering::Relaxed);
            scratch.cached_hits_acc = 0;
        }
        if scratch.pending_hits_acc > 0 {
            self.shared
                .counters
                .pending_hits
                .fetch_add(scratch.pending_hits_acc, Ordering::Relaxed);
            scratch.pending_hits_acc = 0;
        }
        if scratch.query_page_hits_acc > 0 {
            self.shared
                .counters
                .query_page_hits
                .fetch_add(scratch.query_page_hits_acc, Ordering::Relaxed);
            scratch.query_page_hits_acc = 0;
        }
        scratch.last = None;
        scratch.last_page_idx = None;
        scratch.batch_pages.clear();
        scratch.slot_mappings.clear();
        scratch.multi_page_indices.clear();
    }

    #[inline]
    fn load_page_unpinned(&self, page: u64, scratch: &mut DirectScratch) -> Result<Arc<Page>> {
        if let Some((last, data)) = &scratch.last
            && *last == page
        {
            self.shared
                .counters
                .query_page_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(data));
        }
        // 等待前释放上一页与批内固定页句柄，确保阻塞线程不占用缓冲池租约。
        scratch.release_pinned_pages();
        match self
            .shared
            .request(page, false)?
            .ok_or(Error::CorruptIndex("需求读取未取得页句柄"))?
        {
            Entry::Ready(data) => Ok(data),
            Entry::Loading(pending) => pending.wait(),
        }
    }

    pub(crate) fn record<'a>(
        &self,
        slot: usize,
        stride: usize,
        scratch: &'a mut DirectScratch,
    ) -> Result<&'a [u8]> {
        if stride == 0 {
            return Ok(&[]);
        }
        let (start, end) = self.range(slot, stride)?;
        self.shared.check_error()?;
        let p0 = (start / PAGE) as u64;
        let p1 = ((end - 1) / PAGE) as u64;
        if p0 == p1 {
            let data = self.load_page_unpinned(p0, scratch)?;
            data.consume();
            let page_start = p0 as usize * PAGE;
            let lo = start - page_start;
            let hi = end - page_start;
            if hi > data.valid {
                return Err(Error::CorruptIndex("SSD 页不包含完整记录"));
            }
            scratch.last = Some((p0, data));
            return Ok(&scratch.last.as_ref().unwrap().1.bytes()[lo..hi]);
        }
        scratch.bytes.clear();
        scratch.bytes.reserve(stride);
        for page in p0..=p1 {
            let data = self.load_page_unpinned(page, scratch)?;
            data.consume();
            let page_start = page as usize * PAGE;
            let lo = start.saturating_sub(page_start);
            let hi = (end - page_start).min(PAGE);
            if hi > data.valid {
                return Err(Error::CorruptIndex("SSD 页不包含完整记录"));
            }
            scratch.bytes.extend_from_slice(&data.bytes()[lo..hi]);
            scratch.last = Some((page, data));
        }
        Ok(&scratch.bytes)
    }

    pub(crate) fn wait_idle(&self) {
        let mut queue = self.shared.queue.lock().unwrap();
        while queue.active > 0 || !queue.tasks.is_empty() {
            queue = self.shared.idle.wait(queue).unwrap();
        }
    }

    pub(crate) fn clear(&self) {
        self.wait_idle();
        for shard in &self.shared.caches {
            let mut cache = shard.write().unwrap();
            cache.entries.clear();
            cache.fifo.clear();
            cache.resident = 0;
        }
        self.shared
            .counters
            .cached_pages
            .store(0, Ordering::Relaxed);
    }

    pub(crate) fn stats(&self) -> SsdIoStats {
        let c = &self.shared.counters;
        let pool = self.shared.pool.state.lock().unwrap();
        SsdIoStats {
            driver: self.config.driver,
            io_threads: self.workers.len(),
            queue_depth: if self.config.driver == SsdIoDriver::IoUring {
                256
            } else {
                0
            },
            pending_peak_bytes: c.pending_peak.load(Ordering::Relaxed) as usize * PAGE,
            prefetch_skipped: c.prefetch_skipped.load(Ordering::Relaxed),
            uring_submitted: c.uring_submitted.load(Ordering::Relaxed),
            uring_completed: c.uring_completed.load(Ordering::Relaxed),
            uring_max_inflight: c.uring_max_inflight.load(Ordering::Relaxed),
            uring_submit_batches: c.uring_submit_batches.load(Ordering::Relaxed),
            uring_completion_batches: c.uring_completion_batches.load(Ordering::Relaxed),
            uring_completion_batch_max: c.uring_completion_batch_max.load(Ordering::Relaxed),
            uring_enter_calls: c.uring_enter_calls.load(Ordering::Relaxed),
            uring_enter_eintr: c.uring_enter_eintr.load(Ordering::Relaxed),
            ring_errors: c.ring_errors.load(Ordering::Relaxed),
            registered_buffer_bytes: c.registered_buffer_bytes.load(Ordering::Relaxed),
            ring_mapping_bytes: c.ring_mapping_bytes.load(Ordering::Relaxed),
            io_wq_affinity_registered: c.ring_capabilities.load(Ordering::Relaxed) & 1 != 0,
            io_wq_workers_registered: c.ring_capabilities.load(Ordering::Relaxed) & 2 != 0,
            read_calls: c.read_calls.load(Ordering::Relaxed),
            read_bytes: c.read_bytes.load(Ordering::Relaxed),
            read_errors: c.read_errors.load(Ordering::Relaxed),
            cache_hits: c.cache_hits.load(Ordering::Relaxed),
            cache_misses: c.cache_misses.load(Ordering::Relaxed),
            pending_hits: c.pending_hits.load(Ordering::Relaxed),
            query_page_hits: c.query_page_hits.load(Ordering::Relaxed),
            prefetch_requests: c.prefetch_requests.load(Ordering::Relaxed),
            prefetch_used: c.prefetch_used.load(Ordering::Relaxed),
            prefetch_wasted: c.prefetch_wasted.load(Ordering::Relaxed),
            evicted_pages: c.evicted_pages.load(Ordering::Relaxed),
            cache_current_bytes: c.cached_pages.load(Ordering::Relaxed) as usize * PAGE,
            cache_peak_bytes: c.peak_cached_pages.load(Ordering::Relaxed) as usize * PAGE,
            cache_capacity_bytes: self.config.cache_bytes,
            pool_allocated_bytes: pool.allocated * PAGE,
            pool_live_bytes: pool.live * PAGE,
            pool_peak_live_bytes: pool.peak * PAGE,
            pool_limit_bytes: self.shared.pool.limit * PAGE,
        }
    }
}
impl Drop for DirectReader {
    fn drop(&mut self) {
        self.shared.shutting_down.store(true, Ordering::Release);
        let queued = {
            let mut queue = self.shared.queue.lock().unwrap();
            queue.shutdown = true;
            queue.tasks.drain(..).collect::<Vec<_>>()
        };
        for task in queued {
            self.shared.reject(
                &task,
                std::io::Error::new(std::io::ErrorKind::Interrupted, "SSD读取后端已关闭").into(),
            );
        }
        self.shared.available.notify_all();
        {
            let _pool = self.shared.pool.state.lock().unwrap();
            self.shared.pool.ready.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn open_direct(source: &File) -> Result<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
        // Linux UAPI 的 O_DIRECT 标志，重新打开原文件身份以避免修改其他句柄的打开标志。
        const O_DIRECT: i32 = 0o40000;
        Ok(std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(format!("/proc/self/fd/{}", source.as_raw_fd()))?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = source;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "直接 SSD 页读取后端当前支持 Linux，其他平台可使用 mmap",
        )
        .into())
    }
}

fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(bytes, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(bytes, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, bytes, offset);
        Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[cfg(target_os = "linux")]
    #[test]
    fn 直接读取的跨页记录与未对齐文件尾部保持一致() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0..2 * PAGE + 123)
            .map(|index| (index % 239) as u8)
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        file.as_file().sync_all().unwrap();
        let offset = PAGE - 7;
        let length = bytes.len() - offset;
        let reader = DirectReader::open(
            file.as_file(),
            offset,
            length,
            SsdCacheConfig {
                cache_bytes: 16 * PAGE,
                inflight_bytes: 4 * PAGE,
                io_workers: 2,
                prefetch_window: 0,
                driver: SsdIoDriver::IoUring,
            },
        )
        .unwrap();
        let mut scratch = DirectScratch::default();
        assert_eq!(
            reader.record(0, length, &mut scratch).unwrap(),
            &bytes[offset..]
        );
        assert_eq!(reader.stats().read_calls, 3);
        assert_eq!(reader.stats().read_bytes, bytes.len() as u64);
        assert_eq!(reader.stats().read_errors, 0);
    }

    #[test]
    fn 分页缓存跨页尾页命中及预算与原字节一致() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0..37 * PAGE + 123)
            .map(|index| (index * 73 % 251) as u8)
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        let config = SsdCacheConfig {
            cache_bytes: 16 * PAGE,
            inflight_bytes: 4 * PAGE,
            io_workers: 3,
            prefetch_window: 4,
            driver: SsdIoDriver::Pread,
        };
        let reader = DirectReader::from_file(
            file.as_file().try_clone().unwrap(),
            123,
            bytes.len() - 123,
            config,
        )
        .unwrap();
        for slot in 0..40 {
            let mut scratch = DirectScratch::default();
            reader.prefetch(slot, 3072);
            assert_eq!(
                reader.record(slot, 3072, &mut scratch).unwrap(),
                &bytes[123 + slot * 3072..123 + (slot + 1) * 3072]
            );
        }
        let mut first = DirectScratch::default();
        assert_eq!(
            reader.record(39, 3072, &mut first).unwrap(),
            &bytes[123 + 39 * 3072..123 + 40 * 3072]
        );
        drop(first);
        let mut second = DirectScratch::default();
        reader.record(39, 3072, &mut second).unwrap();
        drop(second);
        let stats = reader.stats();
        assert!(stats.cache_hits > 0);
        assert!(stats.evicted_pages > 0);
        assert!(stats.cache_peak_bytes <= config.cache_bytes);
        assert!(stats.pool_peak_live_bytes <= config.cache_bytes + config.inflight_bytes);
        reader.clear();
        assert_eq!(reader.stats().cache_current_bytes, 0);
        assert_eq!(reader.stats().pool_live_bytes, 0);
    }

    #[test]
    fn 零缓存单页在途池并发读取和预取不会占满互等() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0..32 * PAGE)
            .map(|index| (index % 239) as u8)
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        let reader = DirectReader::from_file(
            file.as_file().try_clone().unwrap(),
            0,
            bytes.len(),
            SsdCacheConfig {
                cache_bytes: 0,
                inflight_bytes: PAGE,
                io_workers: 4,
                prefetch_window: 4,
                driver: SsdIoDriver::Pread,
            },
        )
        .unwrap();
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let reader = &reader;
                let bytes = &bytes;
                scope.spawn(move || {
                    let mut scratch = DirectScratch::default();
                    for index in 0..32 {
                        let slot = (index + worker) % 31;
                        reader.prefetch(slot, 4097);
                        assert_eq!(
                            reader.record(slot, 4097, &mut scratch).unwrap(),
                            &bytes[slot * 4097..(slot + 1) * 4097]
                        );
                    }
                });
            }
        });
        reader.clear();
        assert_eq!(reader.stats().cache_current_bytes, 0);
        assert!(reader.stats().pool_peak_live_bytes <= PAGE);
    }

    #[test]
    fn 读取失败传播到等待者且池租约回收() {
        let directory = tempfile::tempdir().unwrap();
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(directory.path().join("write-only.bin"))
            .unwrap();
        file.write_all(&[0; PAGE]).unwrap();
        let reader = DirectReader::from_file(
            file,
            0,
            PAGE,
            SsdCacheConfig {
                cache_bytes: 0,
                inflight_bytes: PAGE,
                io_workers: 2,
                prefetch_window: 0,
                driver: SsdIoDriver::Pread,
            },
        )
        .unwrap();
        let mut scratch = DirectScratch::default();
        assert!(matches!(
            reader.record(0, 16, &mut scratch),
            Err(Error::Io { .. })
        ));
        assert!(matches!(
            reader.record(0, 16, &mut scratch),
            Err(Error::Io { .. })
        ));
        reader.clear();
        assert_eq!(reader.stats().read_errors, 2);
        assert_eq!(reader.stats().pool_live_bytes, 0);
    }

    #[test]
    fn 小缓存十七页预算下多批次并发预取与回退读取不阻塞() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let stride = 1024;
        let pages = 64;
        let total_bytes = pages * PAGE;
        let bytes = (0..total_bytes)
            .map(|i| ((i * 131 + i / PAGE) % 251) as u8)
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        file.as_file().sync_all().unwrap();

        let reader = DirectReader::from_file(
            file.as_file().try_clone().unwrap(),
            0,
            bytes.len(),
            SsdCacheConfig {
                cache_bytes: 16 * PAGE,
                inflight_bytes: PAGE,
                io_workers: 4,
                prefetch_window: 16,
                driver: SsdIoDriver::Pread,
            },
        )
        .unwrap();

        let workers = 20usize;
        let prepared_barrier = std::sync::Barrier::new(workers + 1);
        let release_barrier = std::sync::Barrier::new(workers + 1);
        std::thread::scope(|scope| {
            for worker in 0..workers {
                let reader = &reader;
                let bytes = &bytes;
                let prepared_barrier = &prepared_barrier;
                let release_barrier = &release_barrier;
                scope.spawn(move || {
                    let mut scratch = DirectScratch::default();
                    let slots = [
                        worker * 4,
                        ((worker + 21) % pages) * 4,
                        ((worker + 43) % pages) * 4 + 1,
                    ];
                    reader.prepare_rerank_batch(&slots, stride, &mut scratch);
                    prepared_barrier.wait();
                    release_barrier.wait();

                    let mut misses = Vec::new();
                    for (pos, &slot) in slots.iter().enumerate() {
                        if reader.is_slot_ready(pos, &scratch) {
                            assert_eq!(
                                reader
                                    .record_prepared(pos, slot, stride, &mut scratch)
                                    .unwrap(),
                                &bytes[slot * stride..(slot + 1) * stride]
                            );
                        } else {
                            misses.push(pos);
                        }
                    }
                    let mut pending = Vec::new();
                    for pos in misses {
                        let slot = slots[pos];
                        if reader.try_upgrade_slot_ready(pos, &mut scratch).unwrap() {
                            assert_eq!(
                                reader
                                    .record_prepared(pos, slot, stride, &mut scratch)
                                    .unwrap(),
                                &bytes[slot * stride..(slot + 1) * stride]
                            );
                        } else {
                            pending.push(pos);
                        }
                    }
                    for pos in pending {
                        let slot = slots[pos];
                        assert_eq!(
                            reader
                                .record_prepared(pos, slot, stride, &mut scratch)
                                .unwrap(),
                            &bytes[slot * stride..(slot + 1) * stride]
                        );
                    }
                    reader.finish_rerank_batch(&mut scratch);

                    // 继续并发冲刷混合命中共用页与回退跨页读取。
                    for round in 0..8usize {
                        let base = (worker * 3 + round * 7) % pages;
                        let mixed_slots = [
                            base * 4,
                            ((base + 19) % pages) * 4 + 2,
                            ((base + 37) % pages) * 4 + 3,
                        ];
                        reader.prepare_rerank_batch(&mixed_slots, stride, &mut scratch);
                        for (pos, &slot) in mixed_slots.iter().enumerate() {
                            assert_eq!(
                                reader
                                    .record_prepared(pos, slot, stride, &mut scratch)
                                    .unwrap(),
                                &bytes[slot * stride..(slot + 1) * stride]
                            );
                        }
                        reader.finish_rerank_batch(&mut scratch);
                    }
                });
            }

            // 全部 20 个并发批次持有 prepare 结果且超过 17 页预算时，独立回退读仍能取得租约。
            prepared_barrier.wait();
            let mut extra = DirectScratch::default();
            let probe_slot = 55 * 4 + 2;
            assert_eq!(
                reader.record(probe_slot, stride, &mut extra).unwrap(),
                &bytes[probe_slot * stride..(probe_slot + 1) * stride]
            );
            drop(extra);
            release_barrier.wait();
        });

        reader.clear();
        assert_eq!(reader.stats().cache_current_bytes, 0);
        assert_eq!(reader.stats().pool_live_bytes, 0);
        assert!(reader.stats().pool_peak_live_bytes <= 17 * PAGE);
    }
}
