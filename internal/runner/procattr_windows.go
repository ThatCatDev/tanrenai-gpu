package runner

import (
	"log/slog"
	"os/exec"
	"sync"
	"syscall"
	"unsafe"
)

var (
	kernel32                     = syscall.NewLazyDLL("kernel32.dll")
	procCreateJobObjectW         = kernel32.NewProc("CreateJobObjectW")
	procSetInformationJobObject  = kernel32.NewProc("SetInformationJobObject")
	procAssignProcessToJobObject = kernel32.NewProc("AssignProcessToJobObject")
	procOpenProcess              = kernel32.NewProc("OpenProcess")
	procCloseHandle              = kernel32.NewProc("CloseHandle")
)

const (
	jobObjectLimitKillOnJobClose = 0x00002000
	processAllAccess             = 0x001F0FFF
)

type jobObjectBasicLimitInformation struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type ioCounters struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

type jobObjectExtendedLimitInformation struct {
	BasicLimitInformation jobObjectBasicLimitInformation
	IoInfo                ioCounters
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

var (
	jobMu      sync.Mutex
	jobHandles = make(map[int]syscall.Handle) // pid -> job handle
)

// setSysProcAttr uses CREATE_NEW_PROCESS_GROUP so os.Interrupt works on Windows.
func setSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
}

// afterStart assigns the process to a Windows Job Object with KILL_ON_JOB_CLOSE.
// When the parent process exits (even on crash), Windows closes all handles,
// which triggers automatic termination of all processes in the job.
func afterStart(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid

	// Create an anonymous job object.
	jobHandle, _, err := procCreateJobObjectW.Call(0, 0)
	if jobHandle == 0 {
		slog.Warn("failed to create job object", "error", err)
		return
	}

	// Set KILL_ON_JOB_CLOSE: when last handle closes, kill all processes in job.
	info := jobObjectExtendedLimitInformation{}
	info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose

	const jobObjectExtendedLimitInformationClass = 9
	ret, _, err := procSetInformationJobObject.Call(
		jobHandle,
		uintptr(jobObjectExtendedLimitInformationClass),
		uintptr(unsafe.Pointer(&info)),
		uintptr(unsafe.Sizeof(info)),
	)
	if ret == 0 {
		slog.Warn("failed to set job object limits", "error", err)
		procCloseHandle.Call(jobHandle)
		return
	}

	// Open a handle to the process (cmd.Process.Pid).
	processHandle, _, err := procOpenProcess.Call(processAllAccess, 0, uintptr(pid))
	if processHandle == 0 {
		slog.Warn("failed to open process handle", "pid", pid, "error", err)
		procCloseHandle.Call(jobHandle)
		return
	}

	// Assign the process to the job.
	ret, _, err = procAssignProcessToJobObject.Call(jobHandle, processHandle)
	procCloseHandle.Call(processHandle) // close process handle either way
	if ret == 0 {
		slog.Warn("failed to assign process to job object", "pid", pid, "error", err)
		procCloseHandle.Call(jobHandle)
		return
	}

	jobMu.Lock()
	jobHandles[pid] = syscall.Handle(jobHandle)
	jobMu.Unlock()

	slog.Info("assigned subprocess to job object (kill-on-close)", "pid", pid)
}

// cleanupProcAttr closes the Job Object handle for the given process.
func cleanupProcAttr(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid
	jobMu.Lock()
	h, ok := jobHandles[pid]
	if ok {
		delete(jobHandles, pid)
	}
	jobMu.Unlock()
	if ok {
		procCloseHandle.Call(uintptr(h))
	}
}
