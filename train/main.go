package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/unixpickle/sgd"
	"github.com/unixpickle/textprint/model"
)

const SampleCount = 10000

func main() {
	var batchSize int
	var stepSize float64
	var validationFrac float64
	var netFile string
	var sampleDir string
	var logInterval int
	var maxLen int

	flag.StringVar(&netFile, "file", "out_net", "model output file")
	flag.StringVar(&sampleDir, "samples", "", "sample directory")
	flag.IntVar(&batchSize, "batch", 1, "batch size")
	flag.IntVar(&logInterval, "logint", 4, "log interval")
	flag.IntVar(&maxLen, "maxlen", 0x400, "max article length")
	flag.Float64Var(&stepSize, "step", 0.0001, "step size")
	flag.Float64Var(&validationFrac, "validation", 0.1, "validation fraction")

	flag.Parse()

	if sampleDir == "" {
		fmt.Fprintln(os.Stderr, "Missing required `samples` flag.")
		flag.PrintDefaults()
		os.Exit(1)
	}

	m := readModel(netFile)

	log.Println("Loading samples...")
	samples, err := model.ReadSamples(sampleDir, maxLen)
	if err != nil {
		die("Failed to read samples:", err)
	}
	validationData, trainingData := samples.Split(validationFrac)

	validation := validationData.SampleSet(SampleCount)
	training := trainingData.SampleSet(SampleCount)

	defer func() {
		encoded, err := m.Serialize()
		if err != nil {
			die("Failed to serialize model:", err)
		}
		if err := ioutil.WriteFile(netFile, encoded, 0755); err != nil {
			die("Failed to write model:", err)
		}
	}()

	log.Println("Training...")
	g := &sgd.RMSProp{Gradienter: m, Resiliency: 0.9}
	var lastBatch sgd.SampleSet
	var idx int
	sgd.SGDMini(g, training, stepSize, batchSize, func(batch sgd.SampleSet) bool {
		if idx%logInterval == 0 {
			var last float64
			if lastBatch != nil {
				last = m.Cost(lastBatch).Output()[0]
			}
			lastBatch = batch.Copy()
			cost := m.Cost(batch).Output()[0]
			sgd.ShuffleSampleSet(validation)
			valCost := m.Cost(validation.Subset(0, batchSize)).Output()[0]
			log.Printf("iter %d: val=%f cost=%f last=%f", idx, valCost, cost, last)
		}
		idx++
		return true
	})

	data, err := m.Serialize()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Serialization failed:", err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(netFile, data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save network:", err)
		os.Exit(1)
	}
}

func readModel(path string) *model.Model {
	var m *model.Model
	netData, err := ioutil.ReadFile(path)
	if os.IsNotExist(err) {
		m = model.NewModel()
		log.Println("Constructed new model.")
	} else if err != nil {
		die("Failed to read model:", err)
	} else {
		m, err = model.DeserializeModel(netData)
		if err != nil {
			die("Failed to deserialize model:", err)
		}
		log.Println("Loaded existing model.")
	}
	return m
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
