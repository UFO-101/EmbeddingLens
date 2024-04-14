"use client";
import React from 'react';
import { Bar } from 'react-chartjs-2';
import { CategoryScale, Chart, LinearScale, BarElement } from "chart.js";

Chart.register(CategoryScale, LinearScale, BarElement);

const BarChart = ({ data }) => {
  return (
    <div>
      <Bar
        data={data}
        options={{
          scales: {
            x: {
              type: 'category'
            },
            y: {
              beginAtZero: true
            }
          }
        }}
      />
    </div>
  );
};

export default BarChart;
