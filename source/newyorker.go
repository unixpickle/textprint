package source

import (
	"crypto/md5"
	"encoding/hex"
	"errors"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/yhat/scrape"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// NewYorker is a Source that fetches data from
// The New Yorker (http://www.newyorker.com).
type NewYorker struct{}

// Authors lists the contributors from the page:
// http://www.newyorker.com/contributors/.
func (_ NewYorker) Authors(stop <-chan struct{}) (<-chan Author, <-chan error) {
	authChan := make(chan Author, 1)
	errChan := make(chan error, 1)

	go func() {
		defer close(authChan)
		defer close(errChan)

		resp, err := http.Get("http://www.newyorker.com/contributors/")
		if err != nil {
			errChan <- err
			return
		}

		parsed, err := html.Parse(resp.Body)
		resp.Body.Close()
		if err != nil {
			errChan <- err
			return
		}

		authors, err := NewYorker{}.pageAuthors(parsed)
		if err != nil {
			errChan <- err
			return
		}

		for _, x := range authors {
			select {
			case <-stop:
				return
			case authChan <- x:
			}
		}
	}()

	return authChan, errChan
}

func (_ NewYorker) pageAuthors(page *html.Node) ([]*newYorkerAuthor, error) {
	// Use a map to remove duplicates, since some authors are
	// listed on the page twice.
	res := map[string]*newYorkerAuthor{}

	items := scrape.FindAll(page, func(n *html.Node) bool {
		return scrape.Attr(n, "itemtype") == "http://schema.org/Person"
	})
	for _, item := range items {
		link, ok := scrape.Find(item, scrape.ByTag(atom.A))
		if !ok {
			return nil, errors.New("no link found for person object")
		}
		u := scrape.Attr(link, "href")
		name := strings.TrimSpace(scrape.Text(link))
		if name == "" {
			return nil, errors.New("no name for person object")
		}
		res[u] = &newYorkerAuthor{name: name, url: u}
	}

	list := make([]*newYorkerAuthor, 0, len(res))
	for _, x := range res {
		list = append(list, x)
	}
	return list, nil
}

type newYorkerAuthor struct {
	name string
	url  string
}

func (n *newYorkerAuthor) Name() string {
	return n.name
}

func (n *newYorkerAuthor) Articles(stop <-chan struct{}) (<-chan Article, <-chan error) {
	artChan := make(chan Article, 1)
	errChan := make(chan error, 1)

	go func() {
		defer close(artChan)
		defer close(errChan)

		// filter out certain types of articles, like podcasts.
		whitelist := regexp.MustCompile(`^https?:\/\/www.newyorker.com\/(` +
			`news|magazine)\/.*$`)

		for idx := 1; true; idx++ {
			select {
			case <-stop:
				return
			default:
			}

			urls, next, err := n.fetchPage(idx)
			if err != nil {
				errChan <- err
				return
			}

			for _, u := range urls {
				if !whitelist.MatchString(u) {
					continue
				}
				select {
				case <-stop:
					return
				case artChan <- &newYorkerArticle{url: u}:
				}
			}

			if !next {
				break
			}
		}
	}()

	return artChan, errChan
}

func (n *newYorkerAuthor) fetchPage(idx int) (urls []string, next bool, err error) {
	resp, err := http.Get(n.url + "/all/" + strconv.Itoa(idx))
	if err != nil {
		return
	}

	parsed, err := html.Parse(resp.Body)
	resp.Body.Close()
	if err != nil {
		return
	}

	max, ok := scrape.Find(parsed, scrape.ById("maxPages"))
	if !ok {
		err = errors.New("no maxPages element")
		return
	}

	maxIdx, err := strconv.Atoi(strings.TrimSpace(scrape.Text(max)))
	if err != nil {
		err = errors.New("invalid maxPages element")
		return
	}

	next = maxIdx > idx

	headlines := scrape.FindAll(parsed, func(n *html.Node) bool {
		return scrape.Attr(n, "itemprop") == "headline"
	})
	for _, headline := range headlines {
		parent := headline.Parent
		if parent.DataAtom == atom.A {
			urls = append(urls, scrape.Attr(parent, "href"))
		}
	}

	return
}

type newYorkerArticle struct {
	url string

	pageLock sync.RWMutex
	page     *html.Node
}

func (n *newYorkerArticle) ID() string {
	hash := md5.Sum([]byte(n.url))
	return strings.ToLower(hex.EncodeToString(hash[:]))
}

func (n *newYorkerArticle) Body() (body string, err error) {
	err = n.withPage(func() error {
		artBody, ok := scrape.Find(n.page, scrape.ById("articleBody"))
		if !ok {
			return errors.New("no articleBody element")
		}
		paragraphs := scrape.FindAll(artBody, scrape.ByTag(atom.P))
		var paraText []string
		for _, p := range paragraphs {
			paraText = append(paraText, strings.TrimSpace(scrape.Text(p)))
		}
		body = strings.Join(paraText, "\n\n")
		return nil
	})
	return
}

func (n *newYorkerArticle) Date() (t time.Time, err error) {
	err = n.withPage(func() error {
		meta, ok := scrape.Find(n.page, func(n *html.Node) bool {
			return scrape.Attr(n, "property") == "article:published_time"
		})
		if !ok {
			return errors.New("no date tag found")
		}
		var err error
		t, err = time.Parse(time.RFC3339, scrape.Attr(meta, "content"))
		return err
	})
	return
}

func (n *newYorkerArticle) withPage(f func() error) error {
	n.pageLock.RLock()
	if n.page == nil {
		n.pageLock.RUnlock()
		if err := n.fetchPage(); err != nil {
			return err
		}
		n.pageLock.RLock()
	}
	defer n.pageLock.RUnlock()
	return f()
}

func (n *newYorkerArticle) fetchPage() error {
	n.pageLock.Lock()
	defer n.pageLock.Unlock()

	if n.page != nil {
		return nil
	}

	resp, err := http.Get(n.url)
	if err != nil {
		return err
	}

	n.page, err = html.Parse(resp.Body)
	resp.Body.Close()

	return err
}
