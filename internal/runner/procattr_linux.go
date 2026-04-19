package runner

import (
	"os/exec"
	"syscall"
)

// setSysProcAttr configures the subprocess to receive SIGTERM when the parent dies.
// This prevents orphaned llama-server processes if the CLI exits without cleanup.
func setSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Pdeathsig: syscall.SIGTERM,
	}
}

// afterStart is a no-op on Linux — Pdeathsig handles cleanup automatically.
func afterStart(cmd *exec.Cmd) {}

// cleanupProcAttr is a no-op on Linux.
func cleanupProcAttr(cmd *exec.Cmd) {}
