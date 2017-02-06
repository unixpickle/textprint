package main

import (
	"flag"
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/textprint/model"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var batchSize int
	var stepSize float64
	var validationFrac float64
	var netFile string
	var sampleDir string
	var logInterval int
	var maxLen int

	flag.StringVar(&netFile, "file", "out_net", "model output file")
	flag.StringVar(&sampleDir, "samples", "", "sample directory")
	flag.IntVar(&batchSize, "batch", 8, "batch size")
	flag.IntVar(&logInterval, "logint", 4, "log interval")
	flag.IntVar(&maxLen, "maxlen", 0x200, "max article length")
	flag.Float64Var(&stepSize, "step", 0.001, "step size")
	flag.Float64Var(&validationFrac, "validation", 0.1, "validation fraction")

	flag.Parse()

	if sampleDir == "" {
		essentials.Die("Missing -samples flag. See -help for more.")
	}

	log.Println("Loading model...")
	m := readModel(netFile)

	log.Println("Loading samples...")
	samples, err := model.ReadSamples(sampleDir, maxLen)
	if err != nil {
		essentials.Die("Failed to read samples:", err)
	}
	validationData, trainingData := samples.Split(validationFrac)

	defer func() {
		if err := serializer.SaveAny(netFile, m); err != nil {
			essentials.Die("Failed to save:", err)
		}
	}()

	log.Println("Training...")
	dummyList := make(anyff.SliceSampleList, batchSize)
	t := &model.Trainer{Model: m, Samples: trainingData}
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     t,
		Gradienter:  t,
		Transformer: &anysgd.Adam{},
		Samples:     dummyList,
		Rater:       anysgd.ConstRater(stepSize),
		StatusFunc: func(b anysgd.Batch) {
			if iter%logInterval == 1 {
				vt := &model.Trainer{Model: m, Samples: validationData}
				batch, _ := vt.Fetch(dummyList)
				valCost := vt.TotalCost(batch)
				log.Printf("iter %d: cost=%v validation=%v", iter, t.LastCost,
					anyvec.Sum(valCost.Output()))
			} else {
				log.Printf("iter %d: cost=%v", iter, t.LastCost)
			}
			iter++
		},
	}
	sgd.Run(rip.NewRIP().Chan())
}

func readModel(path string) *model.Model {
	var m *model.Model
	if err := serializer.LoadAny(path, &m); err != nil {
		log.Println("Creating new model.")
		return model.NewModel(anyvec32.CurrentCreator())
	} else {
		log.Println("Using existing model.")
		return m
	}
}
