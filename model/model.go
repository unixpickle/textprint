package model

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
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
func (m *Model) Fingerprints(s sgd.SampleSet) autofunc.Result {
	inSeqs := make([][]linalg.Vector, 0, s.Len()*2)
	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i).(*Sample)
		for _, str := range sample.Articles[:] {
			b := []byte(*str)
			seq := make([]linalg.Vector, len(b))
			for j, ch := range b {
				seq[j] = make(linalg.Vector, 0x100)
				seq[j][int(ch)] = 1
			}
			inSeqs = append(inSeqs, seq)
		}
	}

	in := seqfunc.ConstResult(inSeqs)
	f := rnn.BlockSeqFunc{B: m.Block}
	out := f.ApplySeqs(in)

	return seqfunc.ConcatLast(out)
}

// Cost generates a cost value for a batch.
func (m *Model) Cost(s sgd.SampleSet) autofunc.Result {
	prints := m.Fingerprints(s)
	return autofunc.Pool(prints, func(prints autofunc.Result) autofunc.Result {
		comparisons := m.Comparer.BatchLearner().Batch(prints, s.Len())
		desired := make(linalg.Vector, s.Len())
		for i := range desired {
			if s.GetSample(i).(*Sample).Same {
				desired[i] = 1
			}
		}
		cost := neuralnet.SigmoidCECost{}.Cost(desired, comparisons)
		return autofunc.Scale(cost, 1/float64(s.Len()))
	})
}

// Gradient computes the cost gradient for a batch.
func (m *Model) Gradient(s sgd.SampleSet) autofunc.Gradient {
	grad := autofunc.NewGradient(m.Parameters())
	m.Cost(s).PropagateGradient([]float64{1}, grad)
	return grad
}

// Parameters returns the parameters of the block and the
// decision network.
func (m *Model) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	res = append(res, m.Block.(sgd.Learner).Parameters()...)
	res = append(res, m.Comparer.Parameters()...)
	return res
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
