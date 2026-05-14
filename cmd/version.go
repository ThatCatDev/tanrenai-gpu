package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai-gpu/internal/buildinfo"
)

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print version information",
	Run: func(cmd *cobra.Command, args []string) {
		_, _ = fmt.Fprintf(os.Stdout, "tanrenai-gpu %s\n", buildinfo.Version)
	},
}

func init() {
	rootCmd.AddCommand(versionCmd)
}
