package source

import "time"

// A Source represents a collection of textual articles
// written by various authors.
type Source interface {
	// Authors produces a stream of authors, which can be
	// stopped early by closing the stop channel.
	Authors(stop <-chan struct{}) (<-chan Author, <-chan error)
}

// An Author identifies a writer from a Source.
type Author interface {
	// Name returns the author's human-readable name.
	Name() string

	// Articles fetches a list of articles for the author.
	Articles(stop <-chan struct{}) (<-chan Article, <-chan error)
}

// An Article is a written piece of text.
type Article interface {
	// ID is a unique ID for the article.
	// It should be a string suitable for a filename,
	// preferably of a uniform length.
	ID() string

	// Body attempts to retrieve the body of the article.
	Body() (string, error)

	// Date attempts to retrieve the date of the publication.
	// If no date is available, it returns the zero time.
	// It only returns an error if it cannot fetch the date
	// even though there may be one (e.g. due to HTTP error).
	Date() (time.Time, error)
}
