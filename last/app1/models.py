from django.db import models
from django.contrib.auth.models import User

class HistoryRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    detection_time = models.DateTimeField(auto_now_add=True)
    image = models.CharField(max_length=255)
    predicted_class = models.CharField(max_length=255)
    gender = models.CharField(max_length=10, default='')  # 添加默认值
    physical_type = models.CharField(max_length=50, default='')  # 添加默认值
    additional_physical = models.TextField(default='')  # 添加默认值
    common_performance = models.TextField(default='')  # 添加默认值
    physical_characteristics = models.TextField(default='')  # 添加默认值
    psychological_characteristics = models.TextField(default='')  # 添加默认值
    disease_tendency = models.TextField(default='')  # 添加默认值
    health_advice = models.TextField(default='')  # 添加默认值
    face_diagnosis = models.JSONField(null=True, blank=True)  # 存储面诊结果（JSON格式）
    tongue_diagnosis = models.JSONField(null=True, blank=True)  # 存储舌诊结果（JSON格式）

    def __str__(self):
        return f"{self.user.username} - {self.detection_time}"

    #创建数据库表   字段的属性