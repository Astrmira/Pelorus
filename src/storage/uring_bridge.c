#define _GNU_SOURCE
#include <errno.h>
#include <linux/io_uring.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

struct qv_info {
    uint64_t mapping_bytes, registered_bytes;
    uint32_t sq_entries, cq_entries, features, flags, affinity_count, capabilities;
};
struct qv_stats {
    uint64_t submitted, completed, enter_calls, enter_eintr, max_inflight, completion_batches,
        completion_batch_max, submit_batches;
};
struct qv_event {
    uint64_t token;
    int32_t result;
    uint32_t flags;
};
/* 与Rust侧的repr(C)记录保持一致。 */
_Static_assert(sizeof(struct qv_info) == 40, "信息记录布局不匹配");
_Static_assert(sizeof(struct qv_stats) == 64, "统计记录布局不匹配");
_Static_assert(sizeof(struct qv_event) == 16, "完成记录布局不匹配");
struct qv_ring {
    int fd, files, buffers;
    struct io_uring_params p;
    void *sq_map, *cq_map;
    struct io_uring_sqe *sqes;
    size_t sq_bytes, cq_bytes, sqe_bytes;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array, *sq_dropped;
    unsigned *cq_head, *cq_tail, *cq_mask, *cq_overflow;
    struct io_uring_cqe *cqes;
    uintptr_t buffer_start;
    size_t buffer_bytes;
    unsigned active;
    struct qv_info info;
    struct qv_stats stats;
};
static unsigned acquire(unsigned *p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static void release(unsigned *p, unsigned v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
static int entered(struct qv_ring *r, unsigned submit, unsigned wait) {
    for (;;) {
        r->stats.enter_calls++;
        int n = (int)syscall(__NR_io_uring_enter, r->fd, submit, wait,
                             wait ? IORING_ENTER_GETEVENTS : 0, NULL, 0);
        if (n >= 0)
            return n;
        if (errno != EINTR)
            return -errno;
        r->stats.enter_eintr++;
    }
}
static int registered(struct qv_ring *r, unsigned op, const void *arg, unsigned n) {
    int result = (int)syscall(__NR_io_uring_register, r->fd, op, arg, n);
    return result < 0 ? -errno : result;
}
static void fatal_ring(const char *message) {
    /* 严重环错误不能证明DMA结束，直接停止服务，禁止释放或复用任何仍在途租约。 */
    ssize_t written = write(STDERR_FILENO, message, strlen(message));
    (void)written;
    _exit(86);
}

/* 单生产者且未启用SQPOLL，enter返回后未消费的SQE可以安全撤回。
 * 已消费的请求必须全部收到CQE，才能回收Rust租约和注销固定缓冲。 */
int qv_ring_quiesce(struct qv_ring *r) {
    int first = 0;
    unsigned failures = 0;
    unsigned head = acquire(r->sq_head), tail = acquire(r->sq_tail);
    unsigned queued = tail - head;
    if (queued > r->active)
        fatal_ring("io_uring排空状态损坏，已停止SSD服务以保护在途内存\n");
    release(r->sq_tail, head);
    r->active -= queued;
    while (r->active) {
        head = acquire(r->cq_head);
        tail = acquire(r->cq_tail);
        if (head == tail) {
            int result = entered(r, 0, 1);
            if (result < 0) {
                if (!first)
                    first = result;
                if (++failures >= 3)
                    fatal_ring("io_uring排空连续失败，已停止SSD服务并保留在途租约\n");
                struct timespec pause = {0, 1000000};
                nanosleep(&pause, NULL);
            } else
                failures = 0;
            continue;
        }
        while (head != tail && r->active) {
            head++;
            r->active--;
            r->stats.completed++;
        }
        release(r->cq_head, head);
    }
    return first;
}
int qv_ring_close(struct qv_ring *r) {
    if (!r)
        return 0;
    int first = 0, result;
    if (r->sq_head)
        first = qv_ring_quiesce(r);
    if (r->buffers && (result = registered(r, IORING_UNREGISTER_BUFFERS, NULL, 0)) < 0 && !first)
        first = result;
    if (r->files && (result = registered(r, IORING_UNREGISTER_FILES, NULL, 0)) < 0 && !first)
        first = result;
    if (r->sqes && r->sqes != MAP_FAILED)
        munmap(r->sqes, r->sqe_bytes);
    if (r->cq_map && r->cq_map != MAP_FAILED && r->cq_map != r->sq_map)
        munmap(r->cq_map, r->cq_bytes);
    if (r->sq_map && r->sq_map != MAP_FAILED)
        munmap(r->sq_map, r->sq_bytes);
    if (r->fd >= 0 && close(r->fd) != 0 && !first)
        first = -errno;
    free(r);
    return first;
}
int qv_ring_open(int file, void *buffer, size_t bytes, unsigned depth, struct qv_ring **out,
                 struct qv_info *info) {
    *out = NULL;
    if (depth != 256 || !buffer || !bytes)
        return -EINVAL;
    struct qv_ring *r = calloc(1, sizeof(*r));
    if (!r)
        return -ENOMEM;
    r->fd = -1;
    int error;
    r->fd = (int)syscall(__NR_io_uring_setup, depth, &r->p);
    if (r->fd < 0) {
        error = -errno;
        goto failed;
    }
    if (r->p.flags || r->p.sq_entries < depth || r->p.cq_entries < depth) {
        error = -EPROTO;
        goto failed;
    }
    r->sq_bytes = r->p.sq_off.array + r->p.sq_entries * sizeof(unsigned);
    r->cq_bytes = r->p.cq_off.cqes + r->p.cq_entries * sizeof(struct io_uring_cqe);
    if (r->p.features & IORING_FEAT_SINGLE_MMAP) {
        if (r->cq_bytes > r->sq_bytes)
            r->sq_bytes = r->cq_bytes;
        r->cq_bytes = r->sq_bytes;
    }
    r->sq_map = mmap(NULL, r->sq_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd,
                     IORING_OFF_SQ_RING);
    if (r->sq_map == MAP_FAILED) {
        error = -errno;
        goto failed;
    }
    r->cq_map = r->sq_map;
    if (!(r->p.features & IORING_FEAT_SINGLE_MMAP)) {
        r->cq_map = mmap(NULL, r->cq_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                         r->fd, IORING_OFF_CQ_RING);
        if (r->cq_map == MAP_FAILED) {
            error = -errno;
            goto failed;
        }
    }
    r->sqe_bytes = r->p.sq_entries * sizeof(struct io_uring_sqe);
    r->sqes = mmap(NULL, r->sqe_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd,
                   IORING_OFF_SQES);
    if (r->sqes == MAP_FAILED) {
        error = -errno;
        goto failed;
    }
    char *sq = r->sq_map, *cq = r->cq_map;
    r->sq_head = (unsigned *)(sq + r->p.sq_off.head);
    r->sq_tail = (unsigned *)(sq + r->p.sq_off.tail);
    r->sq_mask = (unsigned *)(sq + r->p.sq_off.ring_mask);
    r->sq_array = (unsigned *)(sq + r->p.sq_off.array);
    r->sq_dropped = (unsigned *)(sq + r->p.sq_off.dropped);
    r->cq_head = (unsigned *)(cq + r->p.cq_off.head);
    r->cq_tail = (unsigned *)(cq + r->p.cq_off.tail);
    r->cq_mask = (unsigned *)(cq + r->p.cq_off.ring_mask);
    r->cq_overflow = (unsigned *)(cq + r->p.cq_off.overflow);
    r->cqes = (struct io_uring_cqe *)(cq + r->p.cq_off.cqes);
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
        error = -errno;
        goto failed;
    }
    /* 旧内核缺少IOWQ控制注册，仍使用相同的固定缓冲批量读取，并显式报告能力。 */
    unsigned capabilities = 0;
    error = registered(r, IORING_REGISTER_IOWQ_AFF, &affinity, sizeof(affinity));
    if (error >= 0)
        capabilities |= 1;
    else if (error != -EINVAL && error != -ENOSYS && error != -EOPNOTSUPP)
        goto failed;
    unsigned workers[2] = {1, 1};
    error = registered(r, IORING_REGISTER_IOWQ_MAX_WORKERS, workers, 2);
    if (error >= 0)
        capabilities |= 2;
    else if (error != -EINVAL && error != -ENOSYS && error != -EOPNOTSUPP)
        goto failed;
    error = registered(r, IORING_REGISTER_FILES, &file, 1);
    if (error < 0)
        goto failed;
    r->files = 1;
    struct iovec vector = {buffer, bytes};
    error = registered(r, IORING_REGISTER_BUFFERS, &vector, 1);
    if (error < 0)
        goto failed;
    r->buffers = 1;
    r->buffer_start = (uintptr_t)buffer;
    r->buffer_bytes = bytes;
    r->info =
        (struct qv_info){r->sq_bytes + r->sqe_bytes + (r->cq_map == r->sq_map ? 0 : r->cq_bytes),
                         bytes,
                         r->p.sq_entries,
                         r->p.cq_entries,
                         r->p.features,
                         r->p.flags,
                         (capabilities & 1) ? (uint32_t)CPU_COUNT(&affinity) : 0,
                         capabilities};
    *info = r->info;
    *out = r;
    return 0;
failed:
    qv_ring_close(r);
    return error;
}
int qv_ring_push(struct qv_ring *r, uint64_t token, void *buffer, unsigned length,
                 uint64_t offset) {
    uintptr_t address = (uintptr_t)buffer;
    if (address < r->buffer_start || length > r->buffer_bytes ||
        address - r->buffer_start > r->buffer_bytes - length)
        return -EFAULT;
    unsigned tail = acquire(r->sq_tail), head = acquire(r->sq_head);
    if (tail - head >= r->p.sq_entries || r->active >= 256)
        return -EBUSY;
    unsigned index = tail & *r->sq_mask;
    struct io_uring_sqe *sqe = &r->sqes[index];
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ_FIXED;
    sqe->flags = IOSQE_FIXED_FILE;
    sqe->fd = 0;
    sqe->addr = address;
    sqe->len = length;
    sqe->off = offset;
    sqe->buf_index = 0;
    sqe->user_data = token;
    r->sq_array[index] = index;
    release(r->sq_tail, tail + 1);
    r->active++;
    if (r->active > r->stats.max_inflight)
        r->stats.max_inflight = r->active;
    return 0;
}
int qv_ring_submit(struct qv_ring *r) {
    unsigned pending = acquire(r->sq_tail) - acquire(r->sq_head);
    if (pending)
        r->stats.submit_batches++;
    while (pending) {
        int n = entered(r, pending, 0);
        if (n < 0)
            return n;
        if (!n || (unsigned)n > pending)
            return -EIO;
        r->stats.submitted += (unsigned)n;
        pending = acquire(r->sq_tail) - acquire(r->sq_head);
    }
    return 0;
}
int qv_ring_reap(struct qv_ring *r, struct qv_event *events, unsigned capacity) {
    if (acquire(r->sq_dropped) || acquire(r->cq_overflow))
        return -EOVERFLOW;
    unsigned head = acquire(r->cq_head), tail = acquire(r->cq_tail);
    if (head == tail && r->active) {
        int result = entered(r, 0, 1);
        if (result < 0)
            return result;
        tail = acquire(r->cq_tail);
    }
    unsigned count = 0;
    while (head != tail && count < capacity) {
        struct io_uring_cqe c = r->cqes[head & *r->cq_mask];
        if (!r->active)
            return -EPROTO;
        events[count++] = (struct qv_event){c.user_data, c.res, c.flags};
        head++;
        r->active--;
        r->stats.completed++;
    }
    release(r->cq_head, head);
    if (count) {
        r->stats.completion_batches++;
        if (count > r->stats.completion_batch_max)
            r->stats.completion_batch_max = count;
    }
    return (int)count;
}
void qv_ring_stats(struct qv_ring *r, struct qv_stats *stats) {
    *stats = r->stats;
}
