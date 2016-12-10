package model

import (
	"crypto/md5"
	"math"
	"math/rand"
	"sort"
)

// Samples stores a set of articles, collected by author.
type Samples struct {
	Articles    [][]string
	AuthorNames []string
}

// Compare selects two samples by the same author.
func (s Samples) Compare() (string, string) {
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
func (s Samples) Contrast() (string, string) {
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
func (s Samples) Split(leftRatio float64) (left Samples, right Samples) {
	sorter := hashSorter{s: s}
	sort.Sort(&sorter)
	leftCount := int(math.Ceil(float64(len(s.Articles)) * leftRatio))
	left = Samples{
		Articles:    s.Articles[:leftCount],
		AuthorNames: s.AuthorNames[:leftCount],
	}
	right = Samples{
		Articles:    s.Articles[leftCount:],
		AuthorNames: s.AuthorNames[leftCount:],
	}
	return
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
	s Samples
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
