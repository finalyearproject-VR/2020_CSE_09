from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from reg.models import *
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers
import json
import markdown2
import bleach

# Create your views here.


def index(request):
   
    events = Upcoming_events.objects.all()
    return render(request, "index.html", {'events': events})


def login(request):
   
    if request.session.has_key('email'):
        email = request.session['email']
        return redirect(f'/reg/{email}')
    elif request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            if email == cred.email and password == cred.password:
                request.session['email'] = email
                messages.success(request, 'Success')
                return redirect(f'/reg/{cred.email}')
            else:
                messages.success(request, 'Wrong credentials')
                return redirect('/login/')
        except:
            messages.success(request, 'Wrong credentials')
            return redirect('/login/')

    return render(request, "login.html")


def signup(request):
    if request.method == "POST":
        name = request.POST['name']
        institution = request.POST['institution']
        mobno = request.POST['mobno']
        dept = request.POST['department']
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            messages.success(request, 'User Already Exists')
            return redirect("/signup/")
            # new_user = user.objects.create(user_id=user_id,user_name=user_name,password=password)
            # return render(request, "entry.html")
        except:
            # return HttpResponse("")
            new_user = User.objects.create(
                name=name, email=email, password=password, mob_no=mobno, institution=institution, department=dept)
            return redirect('/login/')
    return render(request, "signup.html")





#def workshop(request):
 #   workshops = Workshop.objects.all()
  #  return render(request, "workshop.html", {'workshops': workshops})

#def gallery(request):
 #   return render(request, "gallery.html", {'gallery': gallery})



def registration(request, email):
    user = User.objects.get(pk=email)
    workshops = Workshop.objects.all()
    if request.method == "POST":
        enrolled_type = request.POST['enrolled_type']
        enrolled_for = request.POST['enrolled_for']
        new_enrollment = Enrolled(
            email=user, enrolled_type=enrolled_type, enrolled_for=enrolled_for)
        new_enrollment.save()
        messages.success(request, 'Succesfully Registered')
    return render(request, "registration.html", {'workshops': workshops, 'user': user})


def profile_login(request):
    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            if email == cred.email and password == cred.password:
                messages.success(request, 'Success')
                return redirect(f'/my_profile/{cred.email}')
            else:
                messages.success(request, 'Wrong credentials')
                return redirect('/profile_login/')
        except:
            messages.success(request, 'Wrong credentials')
            return redirect('/profile_login/')
    return render(request, "profile_login.html")


def my_profile(request, email):
    user_cred = User.objects.get(pk=email)
    enrolled_events = Enrolled.objects.filter(email=email)
    return render(request, "my_profile.html", {'enrolled_events': enrolled_events, 'email': email, 'user_cred':user_cred})

#def offline_course(request):
 #   offlinecou = Offline_course.objects.all()
  #  return render(request, "offline_course.html", {'offlinecou': offlinecou})

#def offline_description(request, off_id):
 #   offdetails = Offline_description.objects.filter(off_id=off_id)
  #  for offl in offdetails: 
    #    print(offl)
    #return render(request, "offline_description.html",{'offdetails': offdetails})


def login_offline(request):
    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            if email == cred.email and password == cred.password:
                messages.success(request, 'Success')
                return redirect(f'/reg_offline/{cred.email}')
            else:
                messages.success(request, 'Wrong credentials')
                return redirect('/login_offline/')
        except:
            messages.success(request, 'Wrong credentials')
            return redirect('/login_offline/')

    return render(request, "login_offline.html")



def signup_offline(request):
    if request.method == "POST":
        name = request.POST['name']
        institution = request.POST['institution']
        mobno = request.POST['mobno']
        dept = request.POST['department']
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            messages.success(request, 'User Already Exists')
            return redirect("/signup_offline/")
            # new_user = user.objects.create(user_id=user_id,user_name=user_name,password=password)
            # return render(request, "entry.html")
        except:
            # return HttpResponse("")
            new_user = User.objects.create(
                name=name, email=email, password=password,
                mob_no=mobno, institution=institution, department=dept)
            return redirect('/login_offline/')
    return render(request, "signup_offline.html")


def registration_offlinecourse(request, email):
    user = User.objects.get(pk=email)
    offlinecou = Offline_course.objects.all()
    if request.method == "POST":
        enrolled_type = request.POST['enrolled_type']
        enrolled_for = request.POST['enrolled_for']
        new_enrollment = Enrolled(
            email=user, enrolled_type=enrolled_type, enrolled_for=enrolled_for)
        new_enrollment.save()
        messages.success(request, 'Succesfully Registered')
    return render(request, "registration_offline.html", {'offlinecou': offlinecou, 'user': user})


#def inst(request):
 #   insts = Institutes.objects.all()
  #  return render(request, "institutes.html", {'insts': insts})



def online_branch(request):
    branches =Online_branch.objects.all()
    return render(request, "online_branch.html", {'branches': branches})

def online_course(request, bid):
    ondetails = Online_course.objects.filter(bid=bid)
    return render(request, "online_course.html",{'ondetails': ondetails})

def online_course_video(request, c_id):
    videos = Online_course_video.objects.filter(c_id=c_id)
    for courseid in videos:
        print(courseid)
    return render(request, "videos.html", {'videos':videos} )
    
def registration_onlinecourse(request, email):
    user = User.objects.get(pk=email)
    ondetails = Online_course.objects.all()
    if request.method == "POST":
        enrolled_type = request.POST['enrolled_type']
        enrolled_for = request.POST['enrolled_for']
        new_enrollment = Enrolled(
            email=user, enrolled_type=enrolled_type, enrolled_for=enrolled_for)
        new_enrollment.save()
        messages.success(request, 'Succesfully Registered')
    return render(request, "registration_online.html", {'ondetails': ondetails, 'user': user})


def signup_online(request):
    if request.method == "POST":
        name = request.POST['name']
        institution = request.POST['institution']
        mobno = request.POST['mobno']
        dept = request.POST['department']
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            messages.success(request, 'User Already Exists')
            return redirect("/signup_online/")
            # new_user = user.objects.create(user_id=user_id,user_name=user_name,password=password)
            # return render(request, "entry.html")
        except:
            # return HttpResponse("")
            new_user = User.objects.create(
                name=name, email=email, password=password,
                mob_no=mobno, institution=institution, department=dept)
            return redirect('/login_online/')
    return render(request, "signup_online.html")

def login_online(request):
    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            if email == cred.email and password == cred.password:
                messages.success(request, 'Success')
                return redirect(f'/reg_online/{cred.email}')
            else:
                messages.success(request, 'Wrong credentials')
                return redirect('/login_online/')
        except:
            messages.success(request, 'Wrong credentials')
            return redirect('/login_online/')

    return render(request, "login_online.html")




#def internship(request):
 #   intern = Internship.objects.all()
  #  return render(request, "internship.html", {'intern': intern})

def reg_int(request, email):
    user = User.objects.get(pk=email)
    intern = Internship.objects.all()
    if request.method == "POST":
        enrolled_type = request.POST['enrolled_type']
        enrolled_for = request.POST['enrolled_for']
        new_enrollment = Enrolled(
            email=user, enrolled_type=enrolled_type, enrolled_for=enrolled_for)
        new_enrollment.save()
        messages.success(request, 'Succesfully Registered')
    return render(request, "reg_int.html", {'intern': intern, 'user': user})

def signup_int(request):
    if request.method == "POST":
        name = request.POST['name']
        institution = request.POST['institution']
        mobno = request.POST['mobno']
        dept = request.POST['department']
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            messages.success(request, 'User Already Exists')
            return redirect("/signup_int/")
            # new_user = user.objects.create(user_id=user_id,user_name=user_name,password=password)
            # return render(request, "entry.html")
        except:
            # return HttpResponse("")
            new_user = User.objects.create(
                name=name, email=email, password=password,
                mob_no=mobno, institution=institution, department=dept)
            return redirect('/login_int/')
    return render(request, "signup_int.html")

def login_int(request):
    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            if email == cred.email and password == cred.password:
                messages.success(request, 'Success')
                return redirect(f'/reg_int/{cred.email}')
            else:
                messages.success(request, 'Wrong credentials')
                return redirect('/login_int/')
        except:
            messages.success(request, 'Wrong credentials')
            return redirect('/login_int/')

    return render(request, "login_int.html")


#def mentorship(request):
 #   mentorships = Mentorship.objects.all()
  #  return render(request,"mentorship.html",{'mentorships': mentorships})


def signup_mentorship(request):
    if request.method == "POST":
        name = request.POST['name']
        institution = request.POST['institution']
        mobno = request.POST['mobno']
        dept = request.POST['department']
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            messages.success(request, 'User Already Exists')
            return redirect("/signup_mentorship/")
            # new_user = user.objects.create(user_id=user_id,user_name=user_name,password=password)
            # return render(request, "entry.html")
        except:
            # return HttpResponse("")
            new_user = User.objects.create(
                name=name, email=email, password=password,
                mob_no=mobno, institution=institution, department=dept)
            return redirect('/login_mentorship/')
    return render(request, "signup_mentorship.html")


def login_mentorship(request):
    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']
        try:
            cred = User.objects.get(pk=email)
            if email == cred.email and password == cred.password:
                messages.success(request, 'Success')
                return redirect(f'/reg_mentorship/{cred.email}')

            else:
                messages.success(request, 'Wrong credentials')
                return redirect('/login_mentorship/')
        except:
            messages.success(request, 'Wrong credentials')
            return redirect('/login_mentorship/')

    return render(request, "login_mentorship.html")


def registration_mentorship(request, email):
    user = User.objects.get(pk=email)
    mentorships = Mentorship.objects.all()
    if request.method == "POST":
        enrolled_type = request.POST['enrolled_type']
        enrolled_for = request.POST['enrolled_for']
        new_enrollment = Enrolled(
            email=user, enrolled_type=enrolled_type, enrolled_for=enrolled_for)
        new_enrollment.save()
        messages.success(request, 'Succesfully Registered')
    return render(request, "registration_mentorship.html", {'mentorships': mentorships, 'user': user})


def my_profile(request, email):
    user_cred = User.objects.get(pk=email)
    enrolled_events = Enrolled.objects.filter(email=email)
    return render(request, "my_profile.html", {'enrolled_events': enrolled_events, 'email': email, 'user_cred': user_cred})


def newsletter(request):
    newsletters = Newsletter.objects.all()
     
    return render(request, "news_letter.html", {'newsletters': newsletters})

def  Newsletter_videos1(request, newsletter_id):
    videos1 = Newsletter_videos1.objects.filter(newsletter_id=newsletter_id)
    for newsletter_id in videos1:
        print(newsletter_id)
    return render(request, "news_letter.html", {'videos1':videos1} )
    




#def test_categories(request):
 #   test_categories = Test_categories.objects.all()
  #  return render(request, "test_categories.html", {'test_categories': test_categories})


#def test_dept(request, category_id):
 #   test_categories = Test_categories.objects.get(pk=category_id)
  #  depts = Categories_dept.objects.filter(category_id=category_id)
   # return render(request, "test_dept.html", {'depts': depts, 'test_categories': test_categories})


#def test_series(request, dept_id):
 #   tests = Test_series.objects.filter(dept_id=dept_id)
  #  depts = Categories_dept.objects.get(pk=dept_id)
   # return render(request, "test_series.html", {'tests': tests, 'depts': depts})


#def sample_test(request, test_id):
 ##  test_series_objects = Sample_test.objects.filter(test_id=test_id)
   # test_object = Test_series.objects.get(pk=test_id)
    #dept_id = test_object.dept_id

 #   if request.method == "POST":
  #      for question in test_series_objects:
   #         response = request.POST[f"{question.question_id}"]
    #        question_id = question.question_id
     #       test_response = Sample_test_response.objects.create(
      #          user=user[0], question_id=question, test_series_id=question.test_id, response=response)
       #     test_response.save()

        #    result = Sample_test_result(
         #       user=user[0], test_series_id=question.test_id)
          #  if int(response) == question.right_option:
           # else:
            #    result.marks_obtained += 0
        #result.save()
    #return render(request, "sample_test.html", {'test_series_objects': test_series_objects, 'dept_id': dept_id})

#def main(request):
 #   context = {}
  #  context['questions'] = Question.objects.all()
   # return render(request, 'main.html', context)

#def askquestion(request):
 #   if request.method == 'POST':
  #      try:
   #         title = request.POST.get('title')
    #        question = request.POST.get('question')
     #       posted_by = request.POST.get('posted_by')
      #      q = Question(question_title=title, question_text=question, posted_by=posted_by)
       #     q.save()
        #    return redirect(viewquestion, q.qid, q.slug)
        #except Exception as e:
         #   return render(request, 'ask-question.html', { 'error': 'Something is wrong with the form!' })
   # else:
    #    return render(request, 'ask-question.html', {})

#def viewquestion(request, qid, qslug):
 #   context = {}
  #  question = Question.objects.get(qid=qid, slug=qslug)

    # assuming obj is a model instance
   # question_json = json.loads(serializers.serialize('json', [ question ]))[0]['fields']
    #question_json['date_posted'] = question.date_posted
    #question_json['qid'] = question.qid
    #question_json['question_text'] = bleach.clean(markdown2.markdown(question_json['question_text']), tags=['p', 'pre','code', 'sup', 'strong', 'hr', 'sub', 'a'])
    #context['question'] = question_json
    #context['answers'] = []
    #answers = Answer.objects.filter(qid=qid)
    #for answer in answers:
     #   answer.answer_text = bleach.clean(markdown2.markdown(answer.answer_text), tags=['p', 'pre','code', 'sup', 'strong', 'hr', 'sub', 'a'])
      #  context['answers'].append(answer)
    #return render(request, 'view-question.html', context)

#@csrf_exempt
#def ajaxanswerquestion(request):
 #   if request.method == 'POST':
  #      try:
   #         json_data = json.loads(request.body)
    ##       posted_by = json_data['posted_by']
      #      qid = json_data['qid']
       #     a = Answer(answer_text=answer, posted_by=posted_by, qid=Question.objects.get(qid=qid))
        #    a.save()
         #   return JsonResponse({'Success': 'Answer posted successfully.'})
        #except Exception as e:
         #   print(e)
          #  return JsonResponse({'Error': 'Something went wrong when posting your answer.'})
            #return render(request, 'ask-question.html', { 'error': 'Something is wrong with the form!' })





