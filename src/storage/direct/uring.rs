//! 沿用QuIVer的单线程ring所有权、固定缓冲与批量完成回收协议。

use super::*;
use std::{ffi::c_void, os::fd::AsRawFd, ptr::NonNull};

const DEPTH: usize = 256;

#[repr(C)]
#[derive(Default)]
struct Info {
    mapping_bytes: u64,
    registered_bytes: u64,
    sq_entries: u32,
    cq_entries: u32,
    features: u32,
    flags: u32,
    affinity_count: u32,
    capabilities: u32,
}

#[repr(C)]
#[derive(Default)]
struct Stats {
    submitted: u64,
    completed: u64,
    enter_calls: u64,
    enter_eintr: u64,
    max_inflight: u64,
    completion_batches: u64,
    completion_batch_max: u64,
    submit_batches: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Event {
    token: u64,
    result: i32,
    flags: u32,
}

// Rust与C桥接共享固定布局，编译时核对跨语言记录长度。
const _: () = assert!(std::mem::size_of::<Info>() == 40);
const _: () = assert!(std::mem::size_of::<Stats>() == 64);
const _: () = assert!(std::mem::size_of::<Event>() == 16);

unsafe extern "C" {
    fn qv_ring_open(
        file: i32,
        buffer: *mut c_void,
        bytes: usize,
        depth: u32,
        ring: *mut *mut c_void,
        info: *mut Info,
    ) -> i32;
    fn qv_ring_push(
        ring: *mut c_void,
        token: u64,
        buffer: *mut c_void,
        length: u32,
        offset: u64,
    ) -> i32;
    fn qv_ring_enable_coalescing(ring: *mut c_void) -> i32;
    fn qv_ring_submit(ring: *mut c_void) -> i32;
    fn qv_ring_reap(ring: *mut c_void, events: *mut Event, capacity: u32) -> i32;
    fn qv_ring_stats(ring: *mut c_void, stats: *mut Stats);
    fn qv_ring_quiesce(ring: *mut c_void) -> i32;
    fn qv_ring_close(ring: *mut c_void) -> i32;
}

fn checked(result: i32, operation: &str) -> Result<i32> {
    if result < 0 {
        let error = std::io::Error::from_raw_os_error(-result);
        let hint = if error.kind() == std::io::ErrorKind::OutOfMemory {
            "（固定缓冲注册受 RLIMIT_MEMLOCK 限制，可提高 ulimit -l、调小缓存或改用 Pread 驱动）"
        } else {
            ""
        };
        return Err(std::io::Error::new(
            error.kind(),
            format!("io_uring{operation}失败{hint}，系统错误{error}"),
        )
        .into());
    }
    Ok(result)
}

pub(super) struct Ring {
    raw: NonNull<c_void>,
    pool: Arc<Pool>,
}

impl Ring {
    pub(super) fn new(shared: &Shared, precision: RerankPrecision) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        let mut info = Info::default();
        // SAFETY：匿名slab地址及长度固定且页对齐。当前调度线程创建并独占ring，注册持续到排空关闭。
        checked(
            unsafe {
                qv_ring_open(
                    shared.file.as_raw_fd(),
                    shared.pool.address.as_ptr().cast(),
                    shared.pool.slab.len(),
                    DEPTH as u32,
                    &mut raw,
                    &mut info,
                )
            },
            "创建及注册",
        )?;
        let raw =
            NonNull::new(raw).ok_or_else(|| std::io::Error::other("io_uring桥接返回空对象"))?;
        shared
            .counters
            .registered_buffer_bytes
            .store(info.registered_bytes, Ordering::Relaxed);
        shared
            .counters
            .ring_mapping_bytes
            .store(info.mapping_bytes, Ordering::Relaxed);
        shared
            .counters
            .ring_capabilities
            .store(u64::from(info.capabilities), Ordering::Relaxed);
        let ring = Self {
            raw,
            pool: Arc::clone(&shared.pool),
        };
        if precision == RerankPrecision::Fp32 {
            // SAFETY：调度线程独占新建的环，首次提交前分配组描述符，失败时由 Drop 释放环。
            checked(
                unsafe { qv_ring_enable_coalescing(ring.raw.as_ptr()) },
                "启用相邻页读取",
            )?;
        }
        Ok(ring)
    }

    fn push(&mut self, token: u64, lease: &mut Lease, offset: u64) -> Result<()> {
        let pointer = self.pool.write_ptr(lease);
        // SAFETY：尚未发布的独占租约覆盖一页，操作表保留租约到对应CQE或错误排空之后。
        checked(
            unsafe {
                qv_ring_push(
                    self.raw.as_ptr(),
                    token,
                    pointer.cast(),
                    PAGE as u32,
                    offset,
                )
            },
            "准备读取",
        )?;
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        // SAFETY：当前线程独占SQ，所有已准备SQE的租约均保存在active表中。
        checked(unsafe { qv_ring_submit(self.raw.as_ptr()) }, "批量提交")?;
        Ok(())
    }

    fn reap(&mut self, events: &mut [Event]) -> Result<usize> {
        // SAFETY：事件数组长度和C布局一致，桥接在CQE可见后复制并推进CQ。
        Ok(checked(
            unsafe {
                qv_ring_reap(
                    self.raw.as_ptr(),
                    events.as_mut_ptr(),
                    events.len().min(DEPTH) as u32,
                )
            },
            "批量回收",
        )? as usize)
    }

    fn quiesce(&mut self) -> Result<()> {
        // SAFETY：已停止产生新SQE，所有租约仍在active表中，C排空返回前绝不复用。
        checked(unsafe { qv_ring_quiesce(self.raw.as_ptr()) }, "错误排空")?;
        Ok(())
    }

    fn publish(&self, shared: &Shared) {
        let mut stats = Stats::default();
        // SAFETY：ring和输出布局均有效，统计由唯一调度线程复制。
        unsafe {
            qv_ring_stats(self.raw.as_ptr(), &mut stats);
        }
        let c = &shared.counters;
        for (target, value) in [
            (&c.read_calls, stats.submitted),
            (&c.uring_submitted, stats.submitted),
            (&c.uring_completed, stats.completed),
            (&c.uring_max_inflight, stats.max_inflight),
            (&c.uring_submit_batches, stats.submit_batches),
            (&c.uring_completion_batches, stats.completion_batches),
            (&c.uring_completion_batch_max, stats.completion_batch_max),
            (&c.uring_enter_calls, stats.enter_calls),
            (&c.uring_enter_eintr, stats.enter_eintr),
        ] {
            target.store(value, Ordering::Relaxed);
        }
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // SAFETY：池的Arc仍存活，C先排空，再注销固定缓冲和文件，最后关闭ring。
        let result = unsafe { qv_ring_close(self.raw.as_ptr()) };
        if result < 0 {
            tracing::error!(
                "关闭io_uring失败，系统错误{}",
                std::io::Error::from_raw_os_error(-result)
            );
        }
    }
}

struct Operation {
    task: Task,
    lease: Lease,
}

impl Shared {
    fn run_ring(&self, ring: &mut Ring, active: &mut HashMap<u64, Operation>) -> Result<()> {
        let mut token = 0u64;
        let mut events = [Event::default(); DEPTH];
        let mut drained: Vec<Task> = Vec::with_capacity(DEPTH);
        loop {
            let room = DEPTH - active.len();
            if room > 0 {
                let mut queue = self.queue.lock().unwrap();
                let take_n = room.min(queue.tasks.len());
                for _ in 0..take_n {
                    if let Some(task) = queue.tasks.pop_front() {
                        queue.active += 1;
                        drained.push(task);
                    }
                }
                drained.reverse();
            }
            while let Some(task) = drained.pop() {
                let Some(mut lease) = self.try_buffer() else {
                    let mut queue = self.queue.lock().unwrap();
                    queue.active -= 1 + drained.len();
                    if queue.shutdown {
                        let rest: Vec<Task> = drained.drain(..).rev().collect();
                        drop(queue);
                        self.reject(
                            &task,
                            std::io::Error::new(
                                std::io::ErrorKind::Interrupted,
                                "SSD读取后端已关闭",
                            )
                            .into(),
                        );
                        for pending_task in rest {
                            self.reject(
                                &pending_task,
                                std::io::Error::new(
                                    std::io::ErrorKind::Interrupted,
                                    "SSD读取后端已关闭",
                                )
                                .into(),
                            );
                        }
                    } else {
                        for pending_task in drained.drain(..) {
                            queue.tasks.push_front(pending_task);
                        }
                        queue.tasks.push_front(task);
                    }
                    break;
                };
                let queued = task
                    .page
                    .checked_mul(PAGE as u64)
                    .zip(token.checked_add(1))
                    .ok_or(Error::CapacityExceeded)
                    .and_then(|(offset, next)| {
                        ring.push(next, &mut lease, offset)?;
                        Ok(next)
                    });
                let next = match queued {
                    Ok(next) => next,
                    Err(error) => {
                        self.pool.put(lease);
                        self.complete(task, Err(error.clone()));
                        // 整波已从共享队列取出；未交给环的等待者也必须唤醒并归还在途额度。
                        for remaining in drained.drain(..) {
                            self.complete(remaining, Err(error.clone()));
                        }
                        return Err(error);
                    }
                };
                token = next;
                active.insert(token, Operation { task, lease });
            }
            ring.submit()?;
            if active.is_empty() {
                let mut queue = self.queue.lock().unwrap();
                while queue.tasks.is_empty() && !queue.shutdown {
                    queue = self.available.wait(queue).unwrap();
                }
                if queue.shutdown {
                    return Ok(());
                }
                drop(queue);
                self.pool.wait(&self.shutting_down);
                continue;
            }
            let count = ring.reap(&mut events)?;
            // 先发布整批内核统计，随后完成页并唤醒查询，排空快照能够核对提交与完成数。
            ring.publish(self);
            for event in &events[..count] {
                let Operation { task, lease } = active
                    .remove(&event.token)
                    .ok_or(Error::CorruptIndex("异步完成标识没有对应租约"))?;
                let prefetched = task.prefetched;
                let offset = task.page as usize * PAGE;
                let expected = PAGE.min(self.file_len.saturating_sub(offset));
                let result = if event.flags != 0 {
                    Err(Error::CorruptIndex("io_uring完成标志异常"))
                } else if event.result < 0 {
                    Err(std::io::Error::from_raw_os_error(-event.result).into())
                } else {
                    self.counters
                        .read_bytes
                        .fetch_add(event.result as u64, Ordering::Relaxed);
                    if event.result as usize != expected {
                        Err(
                            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "异步SSD页短读")
                                .into(),
                        )
                    } else {
                        Ok(event.result as usize)
                    }
                };
                match result {
                    Ok(valid) => self.complete(
                        task,
                        Ok(Arc::new(Page {
                            lease: Some(lease),
                            valid,
                            pool: Arc::clone(&self.pool),
                            counters: Arc::clone(&self.counters),
                            prefetched,
                            consumed: AtomicBool::new(false),
                        })),
                    ),
                    Err(error) => {
                        self.pool.put(lease);
                        self.counters.read_errors.fetch_add(1, Ordering::Relaxed);
                        self.complete(task, Err(error.clone()));
                        return Err(error);
                    }
                }
            }
        }
    }

    pub(super) fn uring_worker(&self, mut ring: Ring) {
        let mut active = HashMap::with_capacity(DEPTH);
        if let Err(error) = self.run_ring(&mut ring, &mut active) {
            self.counters.ring_errors.fetch_add(1, Ordering::Relaxed);
            self.fail_queued(error.clone());
            for operation in active.values() {
                self.reject(&operation.task, error.clone());
            }
            // 等待者先获得错误，租约和在途计数保留到所有内核DMA完成。
            if let Err(drain) = ring.quiesce() {
                tracing::error!("io_uring错误排空记录{drain}");
            }
            ring.publish(self);
            for (_, operation) in active.drain() {
                self.pool.put(operation.lease);
                drop(operation.task);
                self.finish_io();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn 相邻页按编码提交并核对跨组空洞分散缓冲和文件尾() {
        for (precision, submissions) in [(RerankPrecision::Fp32, 3), (RerankPrecision::Sq8, 15)] {
            核对跨组页读取(precision, submissions);
        }
    }

    fn 核对跨组页读取(precision: RerankPrecision, submissions: u64) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0..16 * PAGE + 37)
            .map(|i| (i * 73 % 251) as u8)
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        file.as_file().sync_all().unwrap();
        // 空闲同步调度器只提供同一套文件和池，测试线程独占被测ring的完整生命周期。
        let reader = DirectReader::open(
            file.as_file(),
            0,
            bytes.len(),
            RerankPrecision::Fp32,
            SsdCacheConfig {
                cache_bytes: 0,
                inflight_bytes: 15 * PAGE,
                io_workers: 1,
                prefetch_window: 0,
                driver: SsdIoDriver::Pread,
            },
        )
        .unwrap();
        let mut ring = Ring::new(&reader.shared, precision).unwrap();
        let mut leases = Vec::new();
        let pages: Vec<_> = (0..9).chain((11..=16).rev()).collect();
        for (slot, page) in pages.iter().enumerate() {
            let mut lease = reader.shared.pool.try_take().unwrap();
            ring.push(slot as u64 + 1, &mut lease, (page * PAGE) as u64)
                .unwrap();
            leases.push(Some(lease));
        }
        ring.submit().unwrap();
        let mut completed = 0;
        // 每次最多交付两页，覆盖同一物理完成需要多次取出的情况。
        let mut events = [Event::default(); 2];
        while completed < 15 {
            let count = ring.reap(&mut events).unwrap();
            for event in &events[..count] {
                assert_eq!(event.flags, 0);
                let slot = event.token as usize - 1;
                let start = pages[slot] * PAGE;
                let expected = &bytes[start..(start + PAGE).min(bytes.len())];
                assert_eq!(event.result, expected.len() as i32);
                let lease = leases[slot].take().expect("每页只能完成一次");
                // SAFETY：对应CQE已到达，租约仍独占有效页，读取完成数据后才归还。
                let actual = unsafe {
                    std::slice::from_raw_parts(
                        reader.shared.pool.address.as_ptr().add(lease.index * PAGE),
                        expected.len(),
                    )
                };
                assert_eq!(actual, expected);
                reader.shared.pool.put(lease);
                completed += 1;
            }
        }
        ring.publish(&reader.shared);
        assert_eq!(reader.stats().read_calls, submissions);
        assert_eq!(reader.stats().uring_submitted, submissions);
        assert_eq!(reader.stats().uring_completed, submissions);
        assert_eq!(reader.stats().uring_submit_batches, 1);
        assert_eq!(reader.stats().uring_max_inflight, submissions);
        assert_eq!(reader.stats().pool_live_bytes, 0);
    }

    #[test]
    fn 合并读遇到截断时唤醒尾页等待者并排空全部租约() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len((4 * PAGE) as u64).unwrap();
        let reader = DirectReader::open(
            file.as_file(),
            0,
            4 * PAGE,
            RerankPrecision::Fp32,
            SsdCacheConfig {
                cache_bytes: 16 * PAGE,
                inflight_bytes: 4 * PAGE,
                prefetch_window: 4,
                ..SsdCacheConfig::default()
            },
        )
        .unwrap();
        file.as_file().set_len((PAGE + 37) as u64).unwrap();
        let mut scratch = DirectScratch::default();
        reader.prepare_rerank_batch(&[0, 1, 2, 3], PAGE, &mut scratch);
        assert!(reader.record_prepared(3, 3, PAGE, &mut scratch).is_err());
        drop(scratch);
        reader.clear();
        let stats = reader.stats();
        assert_eq!(stats.read_calls, 1);
        assert_eq!(stats.uring_completed, 1);
        assert_eq!(stats.read_bytes, (PAGE + 37) as u64);
        assert_eq!(stats.read_errors, 1);
        assert_eq!(stats.pool_live_bytes, 0);
    }

    #[test]
    fn 二十四读者在单页预算下跨页读取并遵守背压() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let stride = PAGE + 13;
        let bytes = (0..16 * stride)
            .map(|i| (i * 43 % 251) as u8)
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        file.as_file().sync_all().unwrap();
        let reader = DirectReader::open(
            file.as_file(),
            0,
            bytes.len(),
            RerankPrecision::Fp32,
            SsdCacheConfig {
                cache_bytes: 0,
                inflight_bytes: PAGE,
                io_workers: 1,
                prefetch_window: 8,
                driver: SsdIoDriver::IoUring,
            },
        )
        .unwrap();
        let start = Arc::new(std::sync::Barrier::new(24));
        std::thread::scope(|scope| {
            for thread in 0..24 {
                let reader = &reader;
                let bytes = &bytes;
                let start = Arc::clone(&start);
                scope.spawn(move || {
                    let mut scratch = DirectScratch::default();
                    start.wait();
                    for iteration in 0..16 {
                        let slot = (thread * 7 + iteration * 11) % 16;
                        reader.prefetch(slot, stride);
                        assert_eq!(
                            reader.record(slot, stride, &mut scratch).unwrap(),
                            &bytes[slot * stride..(slot + 1) * stride]
                        );
                    }
                });
            }
        });
        reader.wait_idle();
        let stats = reader.stats();
        assert_eq!(stats.pending_peak_bytes, PAGE);
        assert_eq!(stats.pool_peak_live_bytes, PAGE);
        assert_eq!(stats.pool_live_bytes, 0);
        assert_eq!(stats.read_errors, 0);
        assert_eq!(stats.ring_errors, 0);
        assert_eq!(stats.uring_submitted, stats.uring_completed);
    }

    #[test]
    fn 异步读取失败唤醒全部等待者并在排空后归还租约() {
        let directory = tempfile::tempdir().unwrap();
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(directory.path().join("write-only.bin"))
            .unwrap();
        file.write_all(&vec![0; 8 * PAGE]).unwrap();
        let reader = DirectReader::from_file(
            file,
            0,
            8 * PAGE,
            RerankPrecision::Fp32,
            SsdCacheConfig {
                cache_bytes: 0,
                inflight_bytes: 16 * PAGE,
                io_workers: 1,
                prefetch_window: 8,
                driver: SsdIoDriver::IoUring,
            },
        )
        .unwrap();
        std::thread::scope(|scope| {
            let start = Arc::new(std::sync::Barrier::new(8));
            for page in 0..8 {
                let reader = &reader;
                let start = Arc::clone(&start);
                scope.spawn(move || {
                    start.wait();
                    assert!(matches!(
                        reader.record(page, PAGE, &mut DirectScratch::default()),
                        Err(Error::Io { .. })
                    ));
                });
            }
        });
        reader.wait_idle();
        let stats = reader.stats();
        assert_eq!(stats.pool_live_bytes, 0);
        assert_eq!(stats.uring_submitted, stats.uring_completed);
        assert_eq!(stats.ring_errors, 1);
        assert!(stats.read_errors > 0);
    }

    #[test]
    fn 读者持有页时关闭取消未提交预读且保持页内容() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0..2 * PAGE).map(|i| (i % 239) as u8).collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        file.as_file().sync_all().unwrap();
        let reader = DirectReader::open(
            file.as_file(),
            0,
            bytes.len(),
            RerankPrecision::Fp32,
            SsdCacheConfig {
                cache_bytes: 0,
                inflight_bytes: PAGE,
                io_workers: 1,
                prefetch_window: 1,
                driver: SsdIoDriver::IoUring,
            },
        )
        .unwrap();
        let mut scratch = DirectScratch::default();
        assert_eq!(
            reader.record(0, PAGE, &mut scratch).unwrap(),
            &bytes[..PAGE]
        );
        reader.prefetch(1, PAGE);
        drop(reader);
        assert_eq!(scratch.last.as_ref().unwrap().1.bytes(), &bytes[..PAGE]);
        // 单页直读采用零拷贝页内切片返回，不再额外复制填充 scratch.bytes。
        assert!(scratch.bytes.is_empty());
    }

    #[test]
    fn io_uring小缓存十七页预算高并发批处理与回退读取不死锁() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let stride = 1025;
        let pages = 64usize;
        let total_bytes = pages * PAGE + stride;
        let slots_total = total_bytes / stride;
        let bytes = (0..total_bytes)
            .map(|i| ((i * 97 + i / PAGE) % 251) as u8)
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        file.as_file().sync_all().unwrap();

        let reader = DirectReader::from_file(
            file.as_file().try_clone().unwrap(),
            0,
            bytes.len(),
            RerankPrecision::Fp32,
            SsdCacheConfig {
                cache_bytes: 16 * PAGE,
                inflight_bytes: PAGE,
                io_workers: 1,
                prefetch_window: 16,
                driver: SsdIoDriver::IoUring,
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
                        (worker * 5) % slots_total,
                        ((worker + 17) * 5) % slots_total,
                        ((worker + 31) * 5 + 3) % slots_total,
                    ];
                    reader.prepare_rerank_batch(&slots, stride, &mut scratch);
                    prepared_barrier.wait();
                    release_barrier.wait();

                    for round in 0..10usize {
                        if round > 0 {
                            let s0 = ((worker * 7 + round * 11) * 3) % slots_total;
                            let s1 = ((worker + round * 13 + 23) * 5) % slots_total;
                            let s2 = (s0 + 17) % slots_total;
                            reader.prepare_rerank_batch(&[s0, s1, s2], stride, &mut scratch);
                            let round_slots = [s0, s1, s2];
                            let mut misses = Vec::new();
                            for (pos, &slot) in round_slots.iter().enumerate() {
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
                            for pos in misses {
                                let _ = reader.try_upgrade_slot_ready(pos, &mut scratch).unwrap();
                                let slot = round_slots[pos];
                                assert_eq!(
                                    reader
                                        .record_prepared(pos, slot, stride, &mut scratch)
                                        .unwrap(),
                                    &bytes[slot * stride..(slot + 1) * stride]
                                );
                            }
                        } else {
                            for (pos, &slot) in slots.iter().enumerate() {
                                assert_eq!(
                                    reader
                                        .record_prepared(pos, slot, stride, &mut scratch)
                                        .unwrap(),
                                    &bytes[slot * stride..(slot + 1) * stride]
                                );
                            }
                        }
                        reader.finish_rerank_batch(&mut scratch);
                    }
                });
            }

            prepared_barrier.wait();
            let mut extra = DirectScratch::default();
            let probe_slot = slots_total - 2;
            assert_eq!(
                reader.record(probe_slot, stride, &mut extra).unwrap(),
                &bytes[probe_slot * stride..(probe_slot + 1) * stride]
            );
            drop(extra);
            release_barrier.wait();
        });

        reader.clear();
        let stats = reader.stats();
        assert_eq!(stats.uring_submitted, stats.uring_completed);
        assert_eq!(stats.cache_current_bytes, 0);
        assert_eq!(stats.pool_live_bytes, 0);
        assert!(stats.pool_peak_live_bytes <= 17 * PAGE);
    }
}
