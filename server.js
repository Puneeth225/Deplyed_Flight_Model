const express = require('express');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

app.post('/predict', (req, res) => {
  const inputData = req.body;

  const pythonProcess = spawn('python', ['predict.py', JSON.stringify(inputData)]);
  let result = '';
  let errorOccurred = false;

  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (error) => {
    console.error(`Python Error: ${error.toString()}`);
    errorOccurred = true;
    res.status(500).send({ error: error.toString() });
  });

  pythonProcess.on('close', (code) => {
    if (!errorOccurred && code === 0) {
      res.json(JSON.parse(result));
    } else if (!errorOccurred) {
      res.status(500).send({ error: 'Unexpected Python script failure.' });
    }
  });
});

app.listen(5000, () => {
  console.log('Server is running on port 5000');
});





// {
//   "Airline": "IndiGo",
//   "Source": "Delhi",
//   "Destination": "Bangalore",
//   "Journey_Day": "25",
//   "Journey_Month": "2",
//   "Total_Stops": "2 stops",
//   "Dep_Time": "12:30",
//   "Arrival_Time": "23:40",
//   "Route": "",
//   "Additional_Info": "",
//   "Duration": "7h 10m"
// }