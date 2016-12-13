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
	Block    rnn.Block
	Comparer neuralnet.Network
}

// DeserializeModel deserializes a model.
func DeserializeModel(d []byte) (*Model, error) {
	var res Model
	err := serializer.DeserializeAny(d, &res.Block, &res.Comparer)
	if err != nil {
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
	comp := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  fingerprintSize * 2,
			OutputCount: 200,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  200,
			OutputCount: 150,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  150,
			OutputCount: 1,
		},
	}
	comp.Randomize()
	return &Model{
		Block: rnn.StackedBlock{
			rnn.NewLSTM(0x100, 0x180),
			rnn.NewLSTM(0x180, 0x180),
			rnn.NewNetworkBlock(outNet, 0),
		},
		Comparer: comp,
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

// Cost generates a cost value for a batch.
// The batch should come from (*Samples).Batch().
func (m *Model) Cost(ins []string) autofunc.Result {
	batch := len(ins) / 4
	prints := m.Fingerprints(ins)
	return autofunc.Pool(prints, func(prints autofunc.Result) autofunc.Result {
		comparisons := m.Comparer.BatchLearner().Batch(prints, batch*2)
		distances := autofunc.Split(batch*2, comparisons)
		var cost autofunc.Result
		for i := 0; i < batch; i++ {
			dist := distances[i]
			if cost == nil {
				cost = dist
			} else {
				cost = autofunc.Add(cost, dist)
			}
		}
		for i := batch; i < 2*batch; i++ {
			dist := distances[i]
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
	return serializer.SerializeAny(m.Block, m.Comparer)
}
