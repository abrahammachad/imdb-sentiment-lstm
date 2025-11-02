# Robot-2025.github.ip
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Robot Dice</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      text-align: center;
      padding-top: 50px;
      background: #f0f0f0;
    }
    .robot {
      font-size: 100px;
    }
    .speech-bubble {
      display: inline-block;
      background: #00aaff;
      color: white;
      padding: 15px 25px;
      border-radius: 25px;
      margin-top: 20px;
      position: relative;
      font-size: 24px;
      max-width: 300px;
    }
    .speech-bubble::after {
      content: "";
      position: absolute;
      bottom: -20px;
      left: 30px;
      border-width: 20px 15px 0;
      border-style: solid;
      border-color: #00aaff transparent;
      display: block;
      width: 0;
    }
  </style>
</head>
<body>
  <div class="robot">🤖</div>
  <div class="speech-bubble">Vamos a jugar Free Fire</div>
</body>
</html>
