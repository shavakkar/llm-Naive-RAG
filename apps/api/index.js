const express = require('express');
const app = express();
const port = 4848;
app.get('/', (req, res) => {
   res.send('Hello, Express!');
});
app.listen(port, () => {
   console.log(`Server is running on http://localhost:${port}`);
});