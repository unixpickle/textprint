package model

import (
	"crypto/md5"
	"io/ioutil"
	"math"
	"math/rand"
	"path/filepath"
	"sort"
)

// Samples stores a set of articles, collected by author.
type Samples struct {
	Articles    [][]string
	AuthorNames []string
}

// ReadSamples reads a set of samples from a directory,
// where each sub-directory corresponds to an author and
// each .txt file inside said directory corresponds to an
// article.
//
// Any article longer than maxLen is truncated.
func ReadSamples(dir string, maxLen int) (*Samples, error) {
	listing, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	res := &Samples{}
	for _, item := range listing {
		if !item.IsDir() {
			continue
		}
		dirPath := filepath.Join(dir, item.Name())
		sub, err := ioutil.ReadDir(dirPath)
		if err != nil {
			return nil, err
		}
		var arts []string
		for _, artItem := range sub {
			if filepath.Ext(artItem.Name()) == ".txt" {
				contents, err := ioutil.ReadFile(filepath.Join(dirPath, artItem.Name()))
				if err != nil {
					return nil, err
				}
				if len(contents) > maxLen {
					contents = contents[:maxLen]
				}
				if len(contents) > 0 {
					arts = append(arts, string(contents))
				}
			}
		}
		if len(arts) > 0 {
			res.AuthorNames = append(res.AuthorNames, item.Name())
			res.Articles = append(res.Articles, arts)
		}
	}
	return res, nil
}

// Compare selects two samples by the same author.
func (s *Samples) Compare() (string, string) {
	var comparable [][]string
	for _, x := range s.Articles {
		if len(x) > 1 {
			comparable = append(comparable, x)
		}
	}
	if len(comparable) == 0 {
		panic("no authors with two or more samples")
	}
	samples := comparable[rand.Intn(len(comparable))]
	idx1, idx2 := sampleSeparate(len(samples))
	return samples[idx1], samples[idx2]
}

// Contrast selects two samples by two separate authors.
func (s *Samples) Contrast() (string, string) {
	if len(s.Articles) < 2 {
		panic("need at least two authors to contrast")
	}
	idx1, idx2 := sampleSeparate(len(s.Articles))
	sample1 := s.Articles[idx1][rand.Intn(len(s.Articles[idx1]))]
	sample2 := s.Articles[idx2][rand.Intn(len(s.Articles[idx2]))]
	return sample1, sample2
}

// Split splits the samples up into a validation and
// training set in a deterministic way.
//
// The original Samples s will be modified in the process.
func (s *Samples) Split(leftRatio float64) (left *Samples, right *Samples) {
	sorter := hashSorter{s: s}
	sort.Sort(&sorter)
	leftCount := int(math.Ceil(float64(len(s.Articles)) * leftRatio))
	left = &Samples{
		Articles:    s.Articles[:leftCount],
		AuthorNames: s.AuthorNames[:leftCount],
	}
	right = &Samples{
		Articles:    s.Articles[leftCount:],
		AuthorNames: s.AuthorNames[leftCount:],
	}
	return
}

// Batch generates a set of strings to be used with
// (*Model).Cost().
//
// Every unit in a batch includes a comparison and a
// contrast, so a batch size of n means 4n strings.
func (s *Samples) Batch(batch int) []string {
	ins := make([]string, 0, 4*batch)
	for i := 0; i < batch; i++ {
		s1, s2 := s.Compare()
		ins = append(ins, s1, s2)
	}
	for i := 0; i < batch; i++ {
		s1, s2 := s.Contrast()
		ins = append(ins, s1, s2)
	}
	return ins
}

func sampleSeparate(n int) (int, int) {
	idx1 := rand.Intn(n)
	idx2 := rand.Intn(n)
	for idx1 == idx2 {
		idx2 = rand.Intn(n)
	}
	return idx1, idx2
}

type hashSorter struct {
	s *Samples
}

func (h *hashSorter) Len() int {
	return len(h.s.AuthorNames)
}

func (h *hashSorter) Swap(i, j int) {
	h.s.AuthorNames[i], h.s.AuthorNames[j] = h.s.AuthorNames[j], h.s.AuthorNames[i]
	h.s.Articles[i], h.s.Articles[j] = h.s.Articles[j], h.s.Articles[i]
}

func (h *hashSorter) Less(i, j int) bool {
	hash1 := md5.Sum([]byte(h.s.AuthorNames[i]))
	hash2 := md5.Sum([]byte(h.s.AuthorNames[j]))
	for i, x := range hash1[:] {
		if x < hash2[i] {
			return true
		}
	}
	return false
}
