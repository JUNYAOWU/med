async function run() {
  await faceapi.nets.ssdMobilenetv1.loadFromUri('/app1/models'); // 加载 SSD 模型，替代 MTCNN
  await faceapi.nets.faceLandmark68Net.loadFromUri('/app1/models'); // 加载人脸关键点检测模型
  // await faceapi.nets.faceRecognitionNet.loadFromUri('/app1/models'); // 加载人脸识别模型

  const videoEl = document.getElementById('inputVideo');
  videoEl.width = 640; // 设置视频宽度
  videoEl.height = 480; // 设置视频高度

  navigator.mediaDevices.getUserMedia({ video: {} })
    .then(stream => videoEl.srcObject = stream)
    .catch(err => console.error(err));

  videoEl.addEventListener('play', async () => {
    const canvas = document.getElementById('overlay');
    // 确保 canvas 的大小与 video 元素的大小一致
    const displaySize = { width: videoEl.videoWidth, height: videoEl.videoHeight };
    canvas.width = displaySize.width;  // 设置 canvas 的宽度
    canvas.height = displaySize.height;  // 设置 canvas 的高度
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
      // 使用 SSD 模型进行人脸检测
      const detections = await faceapi.detectAllFaces(videoEl).withFaceLandmarks().withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

      // 绘制检测到的人脸
      faceapi.draw.drawDetections(canvas, resizedDetections);
      
      // 如果检测到人脸，则绘制人脸关键点
      if (resizedDetections.length) {
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
      }
    }, 100);
  });
}

run();
