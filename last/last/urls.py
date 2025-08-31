"""
URL configuration for last project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView
#导入app的view文件
from app1 import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path("admin/", admin.site.urls),
    path('SheXiangTou',views.SheXiangTou),
    path('QuWeiWenDa',views.QuWeiWenDa),
    path('upload/', views.upload_image, name='upload_image'),
    path('主页',views.index,name='home'),
    path('智能辩证.html', TemplateView.as_view(template_name="app1/智能辩证.html"), name='zhineng_bianzhen'),
    path('save-face-data/', views.save_face_data, name='save_face_data'),
    path('load-face-data/', views.load_face_data, name='load_face_data'), 
    path('history/', views.history_view, name='history'), 
    path('final/', views.show_final, name='show_final'),
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('rlsb',views.rlsb,name='face'),
    path('submit', views.submit_result, name='submit_result'),
    path('zhishiku',views.zhishiku)
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

