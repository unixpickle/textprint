package model

import (
	"math/rand"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A Batch is a batch of training samples.
type Batch struct {
	Seqs anyseq.Seq
	Outs anydiff.Res
}

// A Trainer contains the needed information to train a
// model.
// It is intended to be used as a fetcher and a gradienter
// for an anysgd.SGD.
type Trainer struct {
	Model   *Model
	Samples *Samples

	// LastCost is the cost from the previous Gradient call.
	LastCost anyvec.Numeric
}

// Fetch produces a randomized *Batch of samples.
// The s argument is used only to determine the number of
// samples in the batch.
func (t *Trainer) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	c := t.Model.Parameters()[0].Vector.Creator()
	var seqs [][]anyvec.Vector
	var outputs []anyvec.Vector
	for i := 0; i < s.Len(); i++ {
		var a, b string
		compare := rand.Intn(2) == 0
		if compare {
			a, b = t.Samples.Compare()
			outputs = append(outputs, c.MakeVectorData(c.MakeNumericList([]float64{1})))
		} else {
			a, b = t.Samples.Contrast()
			outputs = append(outputs, c.MakeVectorData(c.MakeNumericList([]float64{0})))
		}
		seqs = append(seqs, stringToSeq(c, a), stringToSeq(c, b))
	}
	return &Batch{
		Seqs: anyseq.ConstSeqList(c, seqs),
		Outs: anydiff.NewConst(c.Concat(outputs...)),
	}, nil
}

// TotalCost computes the cost for a *Batch.
func (t *Trainer) TotalCost(sgdBatch anysgd.Batch) anydiff.Res {
	b := sgdBatch.(*Batch)
	n := b.Outs.Output().Len()
	outSeqs := anyrnn.Map(b.Seqs, t.Model.Block)
	outVecs := anyseq.Tail(outSeqs)
	actual := t.Model.Comparer.Apply(outVecs, n)
	return anynet.SigmoidCE{Average: true}.Cost(b.Outs, actual, 1)
}

// Gradient computes the gradient for a *Batch.
func (t *Trainer) Gradient(sgdBatch anysgd.Batch) anydiff.Grad {
	cost := t.TotalCost(sgdBatch)
	t.LastCost = anyvec.Sum(cost.Output())

	grad := anydiff.NewGrad(t.Model.Parameters()...)
	c := cost.Output().Creator()
	u := c.MakeVectorData(c.MakeNumericList([]float64{1}))
	cost.Propagate(u, grad)

	return grad
}

func stringToSeq(c anyvec.Creator, s string) []anyvec.Vector {
	b := []byte(s)
	res := make([]anyvec.Vector, len(b))
	for i, x := range b {
		oneHot := make([]float64, 0x100)
		oneHot[int(x)] = 1
		res[i] = c.MakeVectorData(c.MakeNumericList(oneHot))
	}
	return res
}
