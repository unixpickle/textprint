package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/signal"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/textprint/model"
)

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
	flag.Float64Var(&stepSize, "step", 0.001, "step size")
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
	validation, training := samples.Split(validationFrac)

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

	killed := make(chan struct{})
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)

	go func() {
		_, ok := <-c
		if !ok {
			return
		}
		signal.Stop(c)
		close(c)
		close(killed)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
	}()

	idx := 0
	trans := sgd.Adam{}
	for {
		select {
		case <-killed:
			return
		default:
		}
		cost := m.Cost(batchSize, training)
		if idx%logInterval == 0 {
			valid := m.Cost(batchSize, validation).Output()[0]
			log.Printf("iter %d:\tvalidation=%f\ttraining=%f", idx, valid,
				cost.Output()[0])
		}
		grad := autofunc.NewGradient(m.Block.(sgd.Learner).Parameters())
		cost.PropagateGradient([]float64{1}, grad)
		grad = trans.Transform(grad)

		grad.AddToVars(-stepSize)
		idx++
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
