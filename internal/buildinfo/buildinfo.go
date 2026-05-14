// Package buildinfo carries version metadata that's stamped into the binary
// at build time (see docker/Dockerfile for the ldflags injection).
package buildinfo

// Version is the release identifier — typically `git describe --tags`
// output (e.g. `v1.5.0`, `v1.5.0-3-gabc1234`, or `v1.5.0-3-gabc1234-dirty`).
// Defaults to `dev` for non-docker local builds.
var Version = "dev"
