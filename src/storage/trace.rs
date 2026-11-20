//! 通过查询线程记录读页决策，通过页任务的所有者归因异步完成字节。

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct SsdRequestTrace {
    pub logical_bytes: u64,
    pub ready_hits: u64,
    pub joined_reads: u64,
    pub page_wait_ns: u64,
    pub owned_pages: Vec<u64>,
    completed: Arc<AtomicU64>,
}

impl SsdRequestTrace {
    /// 返回已成功完成的页字节；要与全局计数对账，应先排空后端。
    pub fn completed_bytes(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<SsdRequestTrace>> = const { RefCell::new(None) };
}

struct Restore(Option<SsdRequestTrace>);

impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.replace(self.0.take());
    }
}

/// 跟踪当前线程内一次同步搜索，包括它发起而由后台完成的页读取。
pub fn trace_ssd_io<T>(search: impl FnOnce() -> T) -> (T, SsdRequestTrace) {
    let _restore = Restore(ACTIVE.replace(Some(SsdRequestTrace::default())));
    let result = search();
    (result, ACTIVE.take().unwrap())
}

pub(super) fn update(f: impl FnOnce(&mut SsdRequestTrace)) {
    ACTIVE.with_borrow_mut(|trace| {
        if let Some(trace) = trace {
            f(trace);
        }
    });
}

pub(super) fn claim(page: u64) -> Option<Arc<AtomicU64>> {
    ACTIVE.with_borrow_mut(|trace| {
        let trace = trace.as_mut()?;
        trace.owned_pages.push(page);
        Some(Arc::clone(&trace.completed))
    })
}

pub(super) fn joined(owner: Option<&Arc<AtomicU64>>) {
    update(|trace| {
        if owner.is_none_or(|owner| !Arc::ptr_eq(owner, &trace.completed)) {
            trace.joined_reads += 1;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 请求结束后完成的读取仍归发起者而非等待者() {
        let (owner, first) = trace_ssd_io(|| claim(7).unwrap());
        let (_, second) = trace_ssd_io(|| {
            joined(Some(&owner));
            let other = claim(8).unwrap();
            joined(Some(&other));
            other.fetch_add(1024, Ordering::Relaxed);
        });
        std::thread::spawn(move || owner.fetch_add(4096, Ordering::Relaxed))
            .join()
            .unwrap();
        assert_eq!(first.owned_pages, [7]);
        assert_eq!(first.completed_bytes(), 4096);
        assert_eq!(second.owned_pages, [8]);
        assert_eq!(second.completed_bytes(), 1024);
        assert_eq!(second.joined_reads, 1);
    }
}
