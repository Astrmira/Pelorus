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
    fn consume(&self) {
        if self.prefetched && !self.consumed.swap(true, Ordering::Relaxed) {
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
    result: Mutex<Option<Result<Arc<Page>>>>,
    ready: Condvar,
}
impl Pending {
    fn wait(&self) -> Result<Arc<Page>> {
        let mut state = self.result.lock().unwrap();
        while state.is_none() {
            state = self.ready.wait(state).unwrap();
        }
        state.as_ref().unwrap().clone()
    }
    fn finish(&self, result: Result<Arc<Page>>) {
        *self.result.lock().unwrap() = Some(result);
        self.ready.notify_all();
    }
}

enum Entry {
    Ready(Arc<Page>),
    Loading(Arc<Pending>),
}
struct Cache {
    entries: HashMap<u64, Entry>,
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
        let pending = Arc::new(Pending {
            result: Mutex::new(None),
            ready: Condvar::new(),
        });
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
            let mut evicted = None;
            for shard in &self.caches {
                let mut cache = shard.write().unwrap();
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
}

#[derive(Default)]
pub(crate) struct DirectScratch {
    pub(crate) bytes: Vec<u8>,
    last: Option<(u64, Arc<Page>)>,
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
                    entries: HashMap::new(),
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

    pub(crate) fn record<'a>(
        &self,
        slot: usize,
        stride: usize,
        scratch: &'a mut DirectScratch,
    ) -> Result<&'a [u8]> {
        let (start, end) = self.range(slot, stride)?;
        self.shared.check_error()?;
        scratch.bytes.clear();
        scratch.bytes.reserve(stride);
        for page in start / PAGE..=(end - 1) / PAGE {
            let page = page as u64;
            let data = if let Some((last, data)) = &scratch.last
                && *last == page
            {
                self.shared
                    .counters
                    .query_page_hits
                    .fetch_add(1, Ordering::Relaxed);
                Arc::clone(data)
            } else {
                // 先归还上一页，单个查询等待下一页时不会额外占用池中的旧页。
                scratch.last = None;
                match self
                    .shared
                    .request(page, false)?
                    .ok_or(Error::CorruptIndex("需求读取未取得页句柄"))?
                {
                    Entry::Ready(data) => data,
                    Entry::Loading(pending) => pending.wait()?,
                }
            };
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
}
