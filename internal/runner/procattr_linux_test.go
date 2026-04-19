//go:build linux

package runner

import (
	"os/exec"
	"testing"
)

// TestSetSysProcAttr verifies that setSysProcAttr sets SysProcAttr on the command.
func TestSetSysProcAttr(t *testing.T) {
	cmd := exec.Command("true")
	setSysProcAttr(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("setSysProcAttr should set SysProcAttr")
	}
}

// TestAfterStart verifies afterStart does not panic (no-op on Linux).
func TestAfterStart(t *testing.T) {
	cmd := exec.Command("true")
	afterStart(cmd) // should not panic
}

// TestCleanupProcAttr verifies cleanupProcAttr does not panic (no-op on Linux).
func TestCleanupProcAttr(t *testing.T) {
	cmd := exec.Command("true")
	cleanupProcAttr(cmd) // should not panic
}
