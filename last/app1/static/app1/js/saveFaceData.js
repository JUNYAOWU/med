// 保存面部数据
function saveFaceData(faceData) {
  fetch('/save-face-data/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCsrfToken(), // Django CSRF 保护
    },
    body: JSON.stringify({ faceData }),
  })
    .then(response => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error('Failed to save data');
      }
    })
    .then(data => {
      console.log('Data saved successfully:', data);
    })
    .catch(error => console.error('Error:', error));
}

// 提取 CSRF Token（Django 模板中需要包含 {% csrf_token %}）
function getCsrfToken() {
  const cookieValue = document.cookie
    .split('; ')
    .find(row => row.startsWith('csrftoken='))
    ?.split('=')[1];
  return cookieValue;
}

// 定义 faceData 数据
const faceData = [
  { label: "Person 1", descriptors: [] },
  { label: "Person 2", descriptors: [] },
];

// 调用函数并传入 faceData
saveFaceData(faceData);
