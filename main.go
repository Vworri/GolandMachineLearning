package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"github.com/sajari/regression"

	"github.com/kniren/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	x := CsvToDataframe()
	// fmt.Printf("Num of rows: %d\n Num of Cols:  %d\n Col Names: %v\n\n", x.Nrow(), x.Ncol(), x.Names())
	// fmt.Println(x.Select([]string{"Rooms", "Suburb", "Price"}).Subset([]int{0, 1, 2, 3, 4}))
	// for _, name := range x.Names() {
	// 	if x.Col(name).Type() != "string" {
	// 		CreateHistogram(&x)
	// 	}
	// }
	_, train := TrainingVSTesting(&x)
	r := CreateModel(&train, "Distance", "Price")
	fmt.Printf("\nRegression Formula:\n%v\n\n", r.Formula)
	ValidateModel(r, &train, "Distance", "Price")
}

//recover from panic
func recoverPanic() {
	if r := recover(); r != nil {
		fmt.Println("recovered from ", r)
	}
}

//Open csv file

func CsvToDataframe() dataframe.DataFrame {
	f, fileOpenErr := os.Open("Data/melb_data.csv")
	if fileOpenErr != nil {
		fmt.Println(fileOpenErr.Error())
	}
	melbData := dataframe.ReadCSV(f)
	f.Close()
	return melbData
}

func CreateHistogram(name string, df *dataframe.DataFrame) {
	defer recoverPanic()
	column := df.Col(name).Float()
	nanValues := 0

	plotVals := make(plotter.Values, len(column))
	summaryVals := make([]float64, len(column))
	for i, val := range column {
		if !math.IsNaN(val) {

			plotVals[i] = val
			summaryVals[i] = val
		} else {
			nanValues = nanValues + 1
		}
	}
	fmt.Printf("The number of nan values in %s is %d\n", name, nanValues)
	p, plottingErr := plot.New()
	if plottingErr != nil {
		log.Fatal(plottingErr.Error())
	}
	p.Title.Text = fmt.Sprintf("%s Histgram", name)
	h, HistErr := plotter.NewHist(plotVals, 50)
	if HistErr != nil {
		log.Fatal(HistErr.Error())
	}
	p.Add(h)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "Plots/Hist/"+name+"_hist.png"); err != nil {
		log.Fatal(err)
	}
}

//Create scatter plots for each coloumn

func CreateScatterPlot(Yname string, Xname string, df *dataframe.DataFrame) {
	defer recoverPanic()
	y := df.Col(Yname).Float()
	column := df.Col(Xname).Float()
	pts := make(plotter.XYs, len(column))
	var nanValues int
	for i, val := range column {
		if !math.IsNaN(val) && !math.IsNaN(y[i]) {
			pts[i].X = val
			pts[i].Y = y[i]
		} else {
			nanValues = nanValues + 1
		}

	}
	p, plotErr := plot.New()
	if plotErr != nil {
		panic(plotErr.Error())
	}
	p.X.Label.Text = Xname
	p.X.Label.Text = Yname
	p.Add(plotter.NewGrid())

	s, scatterErr := plotter.NewScatter(pts)
	if scatterErr != nil {
		panic(scatterErr.Error())
	}
	s.GlyphStyle.Radius = vg.Points(3)
	p.Add(s)
	saveErr := p.Save(4*vg.Inch, 4*vg.Inch, "Plots/Scatter/"+Xname+"_scatter.png")
	if saveErr != nil {
		log.Fatal(saveErr.Error())
	}
	fmt.Printf("The number of nan values in %s vs %s  is %d\n", Xname, Yname, nanValues)

}

//Create training vs testing sets
func TrainingVSTesting(df *dataframe.DataFrame) (dataframe.DataFrame, dataframe.DataFrame) {
	trainingNum := (3 * df.Nrow()) / 4
	indxs := make([]int, df.Nrow())
	for i := 0; i < trainingNum; i++ {
		indxs[i] = i
	}
	trainingDf := df.Subset(indxs[0:trainingNum])

	testingDf := df.Subset(indxs[trainingNum:df.Nrow()])
	return testingDf, trainingDf
}

func CreateModel(training *dataframe.DataFrame, hypothosisName string, tartget string) regression.Regression {
	hypothosis := training.Col(hypothosisName).Float()
	targ := training.Col(tartget).Float()
	weight := training.Col("Landsize").Float()
	var r regression.Regression
	r.SetObserved(fmt.Sprintf("%s Progression", tartget))
	r.SetVar(0, "Distance")
	r.SetVar(1, "Landsize")
	for i, guess := range hypothosis {
		if !math.IsNaN(guess) && !math.IsNaN(targ[i]) && !math.IsNaN(weight[i]) {
			r.Train(regression.DataPoint(targ[i], []float64{guess, weight[i]}))
		}
	}
	r.Run()
	return r
}

func ValidateModel(model regression.Regression, testing *dataframe.DataFrame, hypothosisName string, tartget string) {
	defer recoverPanic()
	hypothosis := testing.Col(hypothosisName).Float()
	weight := testing.Col("Landsize").Float()
	targ := testing.Col(tartget).Float()
	var regressionError float64
	for i, hyp := range hypothosis {
		if !math.IsNaN(hyp) && !math.IsNaN(targ[i]) && !math.IsNaN(weight[i]) {
			yPredicted, err := model.Predict([]float64{hyp, weight[i]})
			if err != nil {
				panic(err.Error())
			}
			regressionError += math.Abs(targ[i]-yPredicted) / float64(testing.Nrow())
		}

	}
	fmt.Printf("MAE = %0.2f\n\n", regressionError)
}
