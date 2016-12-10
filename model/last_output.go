package model

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// lastOutputs takes the last output for each sequence and
// concatenates them into one result.
func lastOutputs(in seqfunc.Result) autofunc.Result {
	var joined linalg.Vector
	for _, x := range in.OutputSeqs() {
		joined = append(joined, x[len(x)-1]...)
	}
	return &lastOutputsResult{
		OutVec: joined,
		In:     in,
	}
}

type lastOutputsResult struct {
	OutVec linalg.Vector
	In     seqfunc.Result
}

func (l *lastOutputsResult) Output() linalg.Vector {
	return l.OutVec
}

func (l *lastOutputsResult) Constant(g autofunc.Gradient) bool {
	return false
}

func (l *lastOutputsResult) PropagateGradient(u linalg.Vector, g autofunc.Gradient) {
	up := make([][]linalg.Vector, len(l.In.OutputSeqs()))
	var idx int
	for i, seq := range l.In.OutputSeqs() {
		up[i] = make([]linalg.Vector, len(seq))
		for j, x := range seq[:len(seq)-1] {
			up[i][j] = make(linalg.Vector, len(x))
		}
		last := seq[len(seq)-1]
		up[i][len(seq)-1] = u[idx : idx+len(last)]
		idx += len(last)
	}
	l.In.PropagateGradient(up, g)
}
