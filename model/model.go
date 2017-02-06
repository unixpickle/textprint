package model

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

const fingerprintSize = 0x180

func init() {
	var m Model
	serializer.RegisterTypedDeserializer(m.SerializerType(), DeserializeModel)
}

// A Model produces fingerprints or costs for strings.
type Model struct {
	Block    anyrnn.Block
	Comparer anynet.Net
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
func NewModel(c anyvec.Creator) *Model {
	return &Model{
		Block: anyrnn.Stack{
			anyrnn.NewLSTM(c, 0x100, 0x180).ScaleInWeights(c.MakeNumeric(0x10)),
			anyrnn.NewLSTM(c, 0x180, fingerprintSize),
		},
		Comparer: anynet.Net{
			anynet.NewFC(c, fingerprintSize*2, 0x80),
			anynet.Tanh,
			anynet.NewFC(c, 0x80, 1),
		},
	}
}

// Parameters returns the parameters of the block and the
// decision network.
func (m *Model) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	res = append(res, m.Block.(anynet.Parameterizer).Parameters()...)
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
