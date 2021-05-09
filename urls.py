"""prinston URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from reg.views import *
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('login/', login),
   
    path('signup/', signup),
    
    path('profile_login/', profile_login),
    path('my_profile/<str:email>', my_profile),
    #path('workshop/', workshop),
    path('reg/<str:email>', registration),
    #path('offline_course/', offline_course),
    #path('offline_description/<int:off_id>/', offline_description),
    path('online_branch/', online_branch),
    path('online_course/<int:bid>', online_course),
    path('login_online/', login_online),
    path('signup_online/', signup_online),
 #   path('institutes/', inst),
    path('videos/<int:c_id>/', online_course_video),
    path('reg_offline/<str:email>', registration_offlinecourse),
    path('reg_online/<str:email>', registration_onlinecourse),
  #  path('internship/', internship),
    path('reg_int/<str:email>', reg_int),
    path('login_int/', login_int),
    path('signup_int/', signup_int),
   # path('gallery/', gallery),
    #path('mentorship/',mentorship),
    #path('login_mentorship/',login_mentorship),
    #path('signup_mentorship/',signup_mentorship),
    #path('reg_mentorship/<str:email>',registration_mentorship),
    
    path('newsletter/', newsletter),
  
 path('videos1/<int:newsletter_id>/', newsletter),
 
  
  #  path('test_categories/', test_categories),
   # path('test_dept/<str:category_id>/', test_dept),
    #path('test_series/<str:dept_id>', test_series),
    #path('sample_test/<str:test_id>', sample_test),
    #path('main/', main),
    #path('question/<int:qid>/<slug:qslug>', viewquestion),
    #path('ask-question', askquestion),
    #path('ajax-answer-question', ajaxanswerquestion),
    
 
   
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
