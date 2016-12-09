package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/unixpickle/textprint/source"
)

func main() {
	if len(os.Args) == 3 {
		if os.Args[1] != "help" {
			dieUsage()
		}
		dieHelp(os.Args[2])
	} else if len(os.Args) == 4 || len(os.Args) == 5 {
		if os.Args[1] != "fetch" {
			dieUsage()
		}
		s, ok := source.Sources[os.Args[2]]
		if !ok {
			fmt.Fprintln(os.Stderr, "Unknown source:", os.Args[2])
			os.Exit(1)
		}
		maxArt := -1
		if len(os.Args) == 5 {
			var err error
			maxArt, err = strconv.Atoi(os.Args[4])
			if err != nil {
				fmt.Fprintln(os.Stderr, "Invalid max_art:", os.Args[4])
				os.Exit(1)
			}
		}
		fetchIntoDir(s, os.Args[3], maxArt)
	} else {
		dieUsage()
	}
}

func fetchIntoDir(s source.Source, out string, maxArt int) {
	if _, err := os.Stat(out); os.IsNotExist(err) {
		if err := os.Mkdir(out, 0755); err != nil {
			fmt.Fprintln(os.Stderr, "Failed to create output directory:", err)
			os.Exit(1)
		}
	}

	authors, errChan := s.Authors(nil)
	for author := range authors {
		log.Println("Fetching author:", author.Name())

		authorPath := filepath.Join(out, author.Name())
		if info, err := os.Stat(authorPath); os.IsNotExist(err) {
			if err := os.Mkdir(authorPath, 0755); err != nil {
				fmt.Fprintln(os.Stderr, "Failed to make author dir:", err)
				os.Exit(1)
			}
		} else if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to stat author dir:", err)
			os.Exit(1)
		} else if !info.IsDir() {
			fmt.Fprintln(os.Stderr, "Author dir path already exists.")
			os.Exit(1)
		}

		stopChan := make(chan struct{})
		arts, errChan1 := author.Articles(stopChan)
		for i := 0; i != maxArt; i++ {
			art, ok := <-arts
			if !ok {
				break
			}
			artPath := filepath.Join(authorPath, art.ID()+".txt")
			if _, err := os.Stat(artPath); !os.IsNotExist(err) {
				log.Println("Skipping article:", art.ID())
				continue
			}
			log.Println("Fetching article:", art.ID())
			body, err := art.Body()
			if err != nil {
				// Not a fatal error because some articles might
				// be broken while others are not.
				log.Println("Failed to fetch body:", err)
				continue
			}
			if err := ioutil.WriteFile(artPath, []byte(body), 0755); err != nil {
				fmt.Fprintln(os.Stderr, "Failed to write output:", err)
				os.Exit(1)
			}
			if date, _ := art.Date(); !date.IsZero() {
				os.Chtimes(artPath, date, date)
			}
		}
		close(stopChan)
		if err := <-errChan1; err != nil {
			fmt.Fprintln(os.Stderr, "Error listing articles:", err)
			os.Exit(1)
		}
	}
	if err := <-errChan; err != nil {
		fmt.Fprintln(os.Stderr, "Error listing authors:", err)
		os.Exit(1)
	}
}

func dieUsage() {
	fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "fetch <source> <output_dir> [max_art]")
	fmt.Fprintln(os.Stderr, "      ", os.Args[0], "help <source>")

	var sourceNames []string
	for name := range source.Sources {
		sourceNames = append(sourceNames, name)
	}
	sort.Strings(sourceNames)

	fmt.Fprintln(os.Stderr, "\nSources:")
	for _, name := range sourceNames {
		fmt.Fprintln(os.Stderr, " *", name)
	}
	fmt.Fprintln(os.Stderr)

	os.Exit(1)
}

func dieHelp(name string) {
	s, ok := source.Sources[name]
	if !ok {
		fmt.Fprintln(os.Stderr, "Unknown source:", name)
	} else {
		fmt.Fprintln(os.Stderr, s.Help())
	}
	os.Exit(1)
}
