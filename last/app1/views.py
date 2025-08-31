from django.http import JsonResponse, HttpResponse
import json
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
import base64
from io import BytesIO
from PIL import Image
from django.shortcuts import render, redirect
from django.core.files.base import ContentFile
import os
from django.conf import settings
import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
from .models import HistoryRecord
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .model import ResNet,Residual

# 预测函数
def predict_image(image_path, model, model_path):
    classes = ['Grey-Black', 'Mirror-Approximated', 'Thin-White', 'White-Greasy', 'Yellow-Greasy']
    device = torch.device('cpu')  # 强制使用 CPU

    # 加载模型权重到 CPU
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)  # 确保模型在 CPU 上
    model.eval()  # 设置为评估模式

    # 图像预处理
    image = Image.open(image_path).convert("RGB")
    normalize = transforms.Normalize(mean=[0.486, 0.495, 0.427], std=[0.184, 0.183, 0.195])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    image = test_transform(image).unsqueeze(0).to(device)  # 数据也移到 CPU

    # 推理
    with torch.no_grad():
        output = model(image)
        pred_class = classes[torch.argmax(output).item()]

    return pred_class



# 判断文件是否有效
def is_valid_file(filename):
    # 过滤掉以点开头的文件（如隐藏文件）和 .DS_Store 文件
    return not filename.startswith('.') and filename != '.DS_Store'


def index(request):
    return render(request, 'app1/主页.html')


def SheXiangTou(request):
    return render(request, 'app1/摄像头.html')


from django.shortcuts import render

def QuWeiWenDa(request):
    
    return render(request, 'app1/趣味问答.html')


def rlsb(request):
    return render(request, 'app1/index.html')


def zhineng(request):
    return render(request, 'app1/index.html')

def zhishiku(request):
    return render(request, 'app1/知识库测试版.html')


# 判断文件是否有效
def is_valid_file(filename):
    # 过滤掉以点开头的文件（如隐藏文件）和 .DS_Store 文件
    return not filename.startswith('.') and filename != '.DS_Store'


@csrf_exempt
def upload_image(request):
    print("Received request method:", request.method)
    if request.method == 'POST':
        # 从请求体中解析 JSON 数据
        image = request.POST.get('image')  # 获取图像数据
        if image:
            # 处理 Base64 数据
            image_data = image.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            print("Decoded image bytes length:", len(image_bytes))  # 打印解码后的图像字节长度

            if request.user.is_authenticated:
                username = request.user.username
            else:
                username = "anonymous"

            # 以用户名创建文件夹
            user_upload_dir = os.path.join(settings.MEDIA_ROOT, 'images', username)
            os.makedirs(user_upload_dir, exist_ok=True)

            # 以时间为图片命名
            now = timezone.now()
            filename = os.path.join(user_upload_dir, f"{now.strftime('%Y%m%d%H%M%S')}.png")
            with open(filename, 'wb') as f:
                f.write(image_bytes)
            print("Saved image to:", filename)  # 打印保存图像的文件路径

            # 初始化
            model_path = os.path.join(settings.MODEL_DIR, r'D:\大学里的杂七杂八\竞赛\互联网＋\med_web(1)\med_web\last\shetai\best_model.pth')
            model = ResNet(Residual)
            device = torch.device("cpu")
            model = model.to(device)

            # 加载模型和原型向量


            # 预测新图像
            predicted_class = predict_image(filename, model,  model_path)
            print(f"预测结果: {predicted_class}")

            # 获取问卷数据
            wenjuan_data = request.session.get('WenJuan', {})
            gender = wenjuan_data.get('性别', '未知')
            physical_type = wenjuan_data.get('体质类型', '未检测')
            additional_physical = wenjuan_data.get('兼有体质', [])
            common_performance = wenjuan_data.get('常见表现', '无')
            physical_characteristics = wenjuan_data.get('形体特征', '无')
            psychological_characteristics = wenjuan_data.get('心理特征', '无')
            disease_tendency = wenjuan_data.get('发病倾向', '无')
            health_advice = wenjuan_data.get('调养建议', '无')

            history_record = HistoryRecord(
                user=request.user if request.user.is_authenticated else None,
                detection_time=now,
                image=filename.replace(settings.MEDIA_ROOT, ''),
                predicted_class=predicted_class,
                gender=gender,
                physical_type=physical_type,
                additional_physical=', '.join(additional_physical),
                common_performance=common_performance,
                physical_characteristics=physical_characteristics,
                psychological_characteristics=psychological_characteristics,
                disease_tendency=disease_tendency,
                health_advice=', '.join(health_advice) if isinstance(health_advice, list) else health_advice,
                # 新增两个字段的赋值
                face_diagnosis={'result': predicted_class, 'type': 'White-Greasy'},  # 示例数据（自动转为JSON）
                tongue_diagnosis={'color': 'Pale', 'coating': 'Thick'},  # 示例数据（自动转为JSON）
            )
            history_record.save()
            request.session['predicted_class'] = predicted_class
            return redirect('show_final')
    return HttpResponse("Invalid request")




# 连接数据库版
@login_required
def show_final(request):
    try:
        # 获取当前用户的最新检测记录
        latest_record = HistoryRecord.objects.filter(user=request.user).latest('detection_time')

        # 构建上下文数据（直接从数据库读取）
        context = {
            'user': request.user.username,
            'timestamp': timezone.localtime(latest_record.detection_time).strftime("%Y-%m-%d %H:%M:%S"),
            'predicted_class': latest_record.predicted_class,
            'gender': latest_record.gender,
            'physical_type': latest_record.physical_type,
            'additional_physical': latest_record.additional_physical.split(
                ', ') if latest_record.additional_physical else [],
            'common_performance': latest_record.common_performance,
            'physical_characteristics': latest_record.physical_characteristics,
            'psychological_characteristics': latest_record.psychological_characteristics,
            'disease_tendency': latest_record.disease_tendency,
            'health_advice': latest_record.health_advice,
            # 新增字段
            'face_diagnosis': latest_record.face_diagnosis,  # 面诊结果（JSON字典）
            'tongue_diagnosis': latest_record.tongue_diagnosis  # 舌诊结果（JSON字典）
        }

    except HistoryRecord.DoesNotExist:
        # 若无记录，返回空数据或错误页面
        context = {
            'error': '暂无检测记录',
            'user': request.user.username if request.user.is_authenticated else "匿名用户"
        }
        return render(request, 'app1/error.html', context, status=404)

    print('final数据已从数据库加载')
    return render(request, 'app1/final.html', context)

# 历史记录
def history_view(request):
    history_records = HistoryRecord.objects.filter(user=request.user) if request.user.is_authenticated else []
    context = {
        'history_records': history_records,
        'MEDIA_URL': settings.MEDIA_URL,  # 手动传递 MEDIA_URL
    }
    return render(request, 'app1/history.html', context)



@csrf_exempt  # 禁用 CSRF 验证（仅限调试环境，生产环境需要处理 CSRF 保护）
def save_face_data(request):
    if request.method == 'POST':
        try:
            # 获取请求体中的 JSON 数据
            data = json.loads(request.body)

            # 从数据中提取面部数据
            face_data = data.get('faceData')

            if not face_data:
                return JsonResponse({'message': 'No face data provided'}, status=400)

            # 定义保存数据的文件路径
            file_path = settings.BASE_DIR / 'face_data.json'

            # 尝试读取现有的 face_data.json 文件
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = []

            # 将新数据添加到现有数据中
            existing_data.extend(face_data)

            # 将数据保存到 JSON 文件
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=4)

            return JsonResponse({'message': 'Data saved successfully'})

        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'message': f'Error: {str(e)}'}, status=500)

    return JsonResponse({'message': 'Invalid request method'}, status=405)


# 新增：获取 face_data.json 数据
def load_face_data(request):
    print('load_face_data')
    print(request.GET)
    try:
        label = request.GET.get('label')  # 从请求参数中获取 label
        if not label:
            return JsonResponse({'message': 'No label provided'}, status=400)

        file_path = settings.BASE_DIR / 'face_data.json'

        # 读取 face_data.json 文件内容
        with open(file_path, 'r') as f:
            data = json.load(f)

        user_exists = False
        for item in data:
            if item.get('label') == label:
                user_exists = True
                break

        if user_exists:
            try:
                user = User.objects.get(username=label)
                history_records = HistoryRecord.objects.filter(user=user)
                history_data = []
                for record in history_records:
                    history_data.append({
                        'user': record.user.username if record.user else 'anonymous',
                        'detection_time': record.detection_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'image': settings.MEDIA_URL + record.image.lstrip('/'),
                        'predicted_class': record.predicted_class
                    })
                return JsonResponse(history_data, safe=False)
            except User.DoesNotExist:
                return JsonResponse({'message': f'User {label} not found'}, status=404)
        else:
            return JsonResponse({'message': f'No user with label "{label}" found'}, status=404)

    except FileNotFoundError:
        return JsonResponse({'message': 'No face data found'}, status=404)
    except Exception as e:
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib import messages
from django.contrib.auth.models import User

# 登录
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect('home')
        else:
            messages.error(request, '用户名或密码错误')
    return render(request, 'app1/登录.html')
# 注册
def register_view(request):
    if request.method == 'POST':    
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            new_user = User.objects.create_user(username=username, password=password)
            messages.success(request, '注册成功，请登录！')
            return redirect('login')
        except:
            messages.error(request, '注册失败，请尝试其他用户名或联系管理员。')
    return render(request, 'app1/register.html')  
# 退出登录
def logout_view(request):
    auth_logout(request)
    return redirect('login')

@csrf_exempt
def submit_result(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print(data)
            gender = '男' if data.get('gender') == 0 else '女'
            physical_type = data.get("result", {}).get("physical")
            additional_physical = data.get("result", {}).get("both", [])  # 注意字段名是both
            health_guide = data.get("result", {}).get("healthGuide", [])

            # 提取常见表现、形体特征、心理特征、发病倾向、调养建议
            common_performance = next((item['changjianbiaoxian'] for item in health_guide if item['name'] == physical_type), '无')
            physical_characteristics = next((item['xingtitezheng'] for item in health_guide if item['name'] == physical_type), '无')
            psychological_characteristics = next((item['xinlitezheng'] for item in health_guide if item['name'] == physical_type), '无')
            disease_tendency = next((item['fabingqingxiang'] for item in health_guide if item['name'] == physical_type), '无')
            health_advice = next((item['tiaoyangfangshi'] for item in health_guide if item['name'] == physical_type), '无')

            # 将WenJuan数据保存到session
            request.session['WenJuan'] = {
                '性别': gender,
                '体质类型': physical_type,
                '兼有体质': additional_physical,
                '常见表现': common_performance,
                '形体特征': physical_characteristics,
                '心理特征': psychological_characteristics,
                '发病倾向': disease_tendency,
                '调养建议': health_advice
            }
            print('submit已收到数据')

            return JsonResponse({"status": "success", "redirect": reverse('show_final')})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)
#推荐内容视图
import requests
from django.http import JsonResponse
from django.shortcuts import render

def final(request):
    context = {
        'user': '张三',
        'timestamp': '2023-10-01 12:34:56',
        'predicted_class': '正常',
        'gender': '男',
        'physical_type': '偏阳',
        'additional_physical': ['偏湿'],
        'common_performance': '面色红润',
        'physical_characteristics': '体型偏瘦',
        'psychological_characteristics': '性格开朗',
        'disease_tendency': '容易感冒',
        'health_advice': ['多喝水', '适当运动']
    }
    return render(request, 'app1/final.html', context)

def get_recommendations(request):
    user = request.GET.get('user')
    gender = request.GET.get('gender')
    physical_type = request.GET.get('physical_type')
    additional_physical = request.GET.get('additional_physical')
    common_performance = request.GET.get('common_performance')
    physical_characteristics = request.GET.get('physical_characteristics')
    psychological_characteristics = request.GET.get('psychological_characteristics')
    disease_tendency = request.GET.get('disease_tendency')
    health_advice = request.GET.get('health_advice')

    # 构建请求数据
    data = {
        'user': user,
        'gender': gender,
        'physical_type': physical_type,
        'additional_physical': additional_physical,
        'common_performance': common_performance,
        'physical_characteristics': physical_characteristics,
        'psychological_characteristics': psychological_characteristics,
        'disease_tendency': disease_tendency,
        'health_advice': health_advice
    }

    # 调用大模型接口
    try:
        response = requests.post('http://10.4.16.17:10680/ui/chat/00d4f7cd7438c0d7', json=data)  # 替换为您的大模型接口地址
        response.raise_for_status()  # 检查请求是否成功
        recommended_content = response.json().get('recommended_content', '未获取到推荐内容')
    except requests.exceptions.RequestException as e:
        recommended_content = f'加载推荐内容失败: {str(e)}'

    return JsonResponse({'recommended_content': recommended_content})
