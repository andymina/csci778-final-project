

const ctx = document.querySelector("#chart");


/* 

Begin Chart.js stuff


*/



let budget_years = []
let budget_amounts = []

const data = {
  labels: budget_years,
  datasets: [
    {
      label: "Billions spent",
      data: budget_amounts,
      borderColor: "rgb(255, 99, 132)",
      backgroundColor: "rgba(255, 99, 132, 0.5)",
      pointStyle: "circle",
      pointRadius: 10,
      pointHoverRadius: 15,
    },
  ],
};



const config = {
  type: "line",
  data: data,
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
        y: {
          beginAtZero: true
        }
      },
    plugins: {
      title: {
        display: true,
        text: "Budget",
      },
    },
  },
};


let onlyChart = new Chart(ctx, config);

/* 

Begin parsing of csv file

*/

const budget_raw = `Year,Combined,Inflation,Scaled 1e9
2006,8395603930,10762502328.38967,10.76
2007,8889497116,11082297188.942982,11.08
2008,9749642465,11698810113.570164,11.7
2009,10355681570,12371284407.408005,12.37
2010,10989317635,12908104761.584131,12.91
2011,10871695465,12416619224.108868,12.42
2012,9872147831,11057643531.5475,11.06
2013,9806070807,10802091447.424746,10.8
2014,9579648298,10414917943.627407,10.41
2015,10808355982,11735966941.822704,11.74
2016,11310541312,12150355392.671919,12.15
2017,11958640003,12599949462.419039,12.6
2018,12495438923,12919151662.562117,12.92
2019,13060914785,13284228048.82084,13.28
2020,13165935267,13165935267.0,13.17`

Papa.parse(budget_raw, {
	//download: true,
  header: true,
  worker: true,
	step: function(row) {
		console.log("Row:", row.data);
    budget_years.push(parseFloat(row.data['Year']));
    budget_amounts.push(parseFloat(row.data['Scaled 1e9']));
	},
	complete: function() {
		console.log(budget_years);
		console.log(budget_amounts);
    onlyChart.update()
	}
});
