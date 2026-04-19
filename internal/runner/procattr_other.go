//go:build !linux && !windows

package runner

import "os/exec"

// setSysProcAttr is a no-op on platforms without parent-death signals or job objects.
func setSysProcAttr(cmd *exec.Cmd) {}

// afterStart is a no-op on platforms without job objects.
func afterStart(cmd *exec.Cmd) {}

// cleanupProcAttr is a no-op on platforms without job objects.
func cleanupProcAttr(cmd *exec.Cmd) {}
