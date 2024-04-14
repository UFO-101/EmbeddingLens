import vibesaurus from '../public/vibesaurus.json'
import BarChart from "./chart";
import Link from 'next/link';

type VibeSaurusEntry = [string, number][]

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
<h1 style={{ padding: '20px', fontWeight: 'bold', fontSize: '2rem' }}>Vibesaurus</h1>
      <h2 style={{ padding: '20px'}}>An SAE was trained, here the features whose top 10 activating words all have unique word stems.</h2>
      <Link href="/activation-unique-stems-histogram">
        View Activation by Unique Stems Histogram
      </Link>

<div style={{ overflowX: 'scroll', maxWidth: '100%' }}>
  <div className="flex flex-wrap">
    {Object.keys(vibesaurus).map((key) => {
      const data: VibeSaurusEntry = (vibesaurus as any)[key];

      const barChartData = {
        labels: data.map((e) => e[0]),
        datasets: [
          {
            label: key,
            backgroundColor: 'rgba(75,192,192,1)',
            borderColor: 'rgba(0,0,0,1)',
            borderWidth: 2,
            data: data.map((e) => e[1]),
          },
        ],
      };

      return (
        <div key={key} style={{ marginRight: '20px', marginBottom: '20px', minWidth: '400px' }}>
          <BarChart
            data={barChartData}
            options={{
              plugins: {
                legend: {
                  display: false,
                },
              },
              scales: {
                x: {
                  ticks: {
                    autoSkip: false,
                    fontSize: 10, // Adjust font size as needed
                    maxRotation: 90, // Rotate labels if necessary
                    minRotation: 90,
                  },
                },
              },
            }}
          />
        </div>
      );
    })}
  </div>
</div>

    </main>
  );
}
