package source

import (
	"fmt"
	"testing"
)

func TestNewYorker(t *testing.T) {
	authorChan, errChan := NewYorker{}.Authors(nil)
	var authors []Author
	for a := range authorChan {
		authors = append(authors, a)
	}
	if err := <-errChan; err != nil {
		t.Fatal(err)
	}

	stopChan := make(chan struct{}, 1)
	artChan, errChan := authors[0].Articles(stopChan)
	art1 := <-artChan
	if art1 == nil {
		t.Fatal("no first article:", <-errChan)
	}
	close(stopChan)
	if err := <-errChan; err != nil {
		t.Fatal(err)
	}

	body, err := art1.Body()
	if err != nil {
		t.Error(err)
	} else if body == "" {
		t.Error("empty body")
	}

	fmt.Println(body)

	_, err = art1.Date()
	if err != nil {
		t.Error(err)
	}
}
