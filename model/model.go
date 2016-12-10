package model

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const fingerprintSize = 50

func init() {
	var m Model
	serializer.RegisterTypedDeserializer(m.SerializerType(), DeserializeModel)
}

// A Model produces fingerprints or costs for strings.
type Model struct {
	Block rnn.Block
}

// DeserializeModel deserializes a model.
func DeserializeModel(d []byte) (*Model, error) {
	var res Model
	if err := serializer.DeserializeAny(d, &res.Block); err != nil {
		return nil, err
	}
	return &res, nil
}

// NewModel creates a fresh, untrained model.
func NewModel() *Model {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  0x180,
			OutputCount: fingerprintSize,
		},
		&neuralnet.HyperbolicTangent{},
	}
	outNet.Randomize()
	return &Model{
		Block: rnn.StackedBlock{
			rnn.NewLSTM(0x100, 0x180),
			rnn.NewLSTM(0x180, 0x180),
			rnn.NewNetworkBlock(outNet, 0),
		},
	}
}

// Fingerprints generates a concatenated result with the
// fingerprint of each string in ordered.
func (m *Model) Fingerprints(s []string) autofunc.Result {
	inSeqs := make([][]linalg.Vector, len(s))
	for i, x := range s {
		b := []byte(x)
		inSeqs[i] = make([]linalg.Vector, len(b))
		for j, ch := range b {
			inSeqs[i][j] = make(linalg.Vector, 0x100)
			inSeqs[i][j][int(ch)] = 1
		}
	}

	in := seqfunc.ConstResult(inSeqs)
	f := rnn.BlockSeqFunc{B: m.Block}
	out := f.ApplySeqs(in)

	return lastOutputs(out)
}

// Cost generates a cost value for a batch of training.
// Every unit in a batch includes a comparison and a
// contrast, so a batch size of n means 4n sequences.
func (m *Model) Cost(batch int, s *Samples) autofunc.Result {
	ins := make([]string, 0, 4*batch)
	for i := 0; i < batch; i++ {
		s1, s2 := s.Compare()
		ins = append(ins, s1, s2)
	}
	for i := 0; i < batch; i++ {
		s1, s2 := s.Contrast()
		ins = append(ins, s1, s2)
	}
	prints := m.Fingerprints(ins)
	return autofunc.Pool(prints, func(prints autofunc.Result) autofunc.Result {
		split := autofunc.Split(batch*4, prints)
		var cost autofunc.Result
		for i := 0; i < batch; i++ {
			dist := distance(split[i*2], split[i*2+1])
			if cost == nil {
				cost = dist
			} else {
				cost = autofunc.Add(cost, dist)
			}
		}
		for i := 0; i < batch; i++ {
			dist := distance(split[i*2+batch*2], split[i*2+batch*2+1])
			cost = autofunc.Add(cost, autofunc.Scale(dist, -1))
		}
		return autofunc.Scale(cost, 1/float64(2*batch))
	})
}

// SerializerType returns the unique ID used to serialize
// a Model with the serializer package.
func (m *Model) SerializerType() string {
	return "github.com/unixpickle/textprint/model.Model"
}

// Serialize serializes the model.
func (m *Model) Serialize() ([]byte, error) {
	return serializer.SerializeAny(m.Block)
}

func distance(v1, v2 autofunc.Result) autofunc.Result {
	sub := autofunc.Add(v1, autofunc.Scale(v2, -1))
	mag2 := autofunc.SquaredNorm{}.Apply(sub)
	return autofunc.Pow(mag2, 0.5)
}
