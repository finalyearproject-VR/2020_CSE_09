from django.db import models
from django.utils.text import slugify

# Create your models here.


class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(primary_key=True)
    password = models.CharField(max_length=20)
    mob_no = models.IntegerField()
    institution = models.CharField(max_length=100)
    department = models.CharField(max_length=100)

    def __str__(self):
        return self.name


#class Workshop(models.Model):
 #   workshop_id = models.IntegerField(primary_key=True)
   # title = models.CharField(max_length=30)
    #description = models.CharField(max_length=120)
   # image = models.ImageField(blank=True, upload_to="images/")

    #def __str__(self):
     #   return self.title


 #class Enrolled(models.Model):
   # email = models.ForeignKey(User, on_delete=models.CASCADE)
    #enrolled_type = models.CharField(max_length=100)
    #enrolled_for = models.CharField(max_length=100)

   # def __str__(self):
       # return str(self.email)


class Upcoming_events(models.Model):
    event_id = models.IntegerField(primary_key=True)
    event_name = models.CharField(max_length=100)
    event_desc = models.CharField(max_length=500)
    event_type = models.CharField(max_length=100)
    event_date = models.IntegerField()
    event_month = models.CharField(max_length=100)
    event_image = models.ImageField(blank=True, upload_to="images/")

    def __str__(self):
        return self.event_name
        
#class Offline_course(models.Model):
 #   off_id = models.IntegerField(primary_key=True)
  #  title = models.CharField(max_length=30)
   # description = models.CharField(max_length=300)
    #image = models.ImageField(blank=True, upload_to="images/")
    

    #def __str__(self):
     #   return self.title

#class Offline_description(models.Model):
 #   off_id = models.ForeignKey(Offline_course, on_delete=models.CASCADE)
  #  course = models.CharField(max_length=500)
   # career = models.CharField(max_length=500)
   # syllabus = models.CharField(max_length=500)
   # scope = models.CharField(max_length=500)

   # def __str__(self):
    #    return str(self.off_id)

#class Institutes(models.Model):
 #   insti_id = models.IntegerField(primary_key=True)
  #  title = models.CharField(max_length=30)
   # address = models.CharField(max_length=300)
   # course = models.CharField(max_length=200)
    #image = models.ImageField(blank=True, upload_to="images/")

    #def __str__(self):
     #   return self.title


class Online_branch(models.Model):
    bid = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=30)
    description = models.CharField(max_length=300)
    image = models.ImageField(blank=True, upload_to="images/")
    

    def __str__(self):
        return self.title

class Online_course(models.Model):
    bid = models.ForeignKey(Online_branch, on_delete=models.CASCADE)
    c_id = models.IntegerField(primary_key=True)    #course id
    c_title = models.CharField(max_length=500)        #course title
    c_description = models.CharField(max_length=500)           #course desc
    intro_video = models.FileField(blank=True, upload_to="videos/")

    def __str__(self):
        return str(self.c_title)
 

class Online_course_video(models.Model):
    c_id = models.ForeignKey(Online_course, on_delete=models.CASCADE)
    title = models.CharField(max_length=30)
    video = models.FileField(upload_to="videos/")

    def __str__(self):
        return str(self.c_id)


class Newsletter(models.Model):
    newsletter_id = models.CharField(primary_key=True, max_length=10)
    title = models.CharField(max_length=1000)
    description = models.TextField()
    videos1 = models.FileField(blank=True,upload_to="videos1/")
   
    def __str__(self):
       return self.title
class Newsletter_videos1(models.Model):
    newsletter_id = models.ForeignKey(Newsletter, on_delete=models.CASCADE)
    title = models.CharField(max_length=300)
   
    videos1 = models.FileField(upload_to="videos1/")

    def __str__(self):
        return str(self.newsletter_id)



#class Test_categories(models.Model):
#    category_id = models.CharField(primary_key=True, max_length=10)
 #   category_title = models.CharField(max_length=50)
  #  image = models.ImageField(blank=True, upload_to="images/")
   # link = models.CharField(max_length=25)

    #def __str__(self):
     #   return self.category_title


#class Categories_dept(models.Model):
 #   category_id = models.ForeignKey(Test_categories, on_delete=models.CASCADE)
  #  dept_id = models.CharField(primary_key=True, max_length=10)
   # dept_title = models.CharField(max_length=50)
    #image = models.ImageField(blank=True, upload_to="images/")
    #link = models.CharField(max_length=25)

   # def __str__(self):
    #    return self.dept_id


#class Test_series(models.Model):
 #   dept_id = models.ForeignKey(Categories_dept, on_delete=models.CASCADE)
 #   test_id = models.CharField(primary_key=True, max_length=10)
  #  title = models.CharField(max_length=50)
   # description = models.TextField()
    #image = models.ImageField(blank=True, upload_to="images/")
    #sample_test_link = models.CharField(max_length=25)
    #test_link = models.CharField(max_length=25)

    #def __str__(self):
     #   return self.title


#class Sample_test(models.Model):
 #   test_id = models.ForeignKey(Test_series, on_delete=models.CASCADE)
 #   question_id = models.IntegerField(primary_key=True)
 #   question = models.TextField(max_length=200)
 #   option_1 = models.CharField(max_length=100)
  #  option_2 = models.CharField(max_length=100)
   # option_3 = models.CharField(max_length=100)
   # option_4 = models.CharField(max_length=100)
    #right_option = models.IntegerField()

    #def __str__(self):
     #   return str(self.test_series_id)


#class Sample_test_response(models.Model):
 #   user = models.ForeignKey(User, on_delete=models.CASCADE)
  #  question_id = models.ForeignKey(Sample_test, on_delete=models.CASCADE)
   # test_series_id = models.ForeignKey(Test_series, on_delete=models.CASCADE)
    #response = models.IntegerField()

    #def __str__(self):
     #   return str(self.user)


#class Sample_test_result(models.Model):
 #   user = models.ForeignKey(User, on_delete=models.CASCADE)
  #  test_series_id = models.ForeignKey(Test_series, on_delete=models.CASCADE)
   # max_marks = models.IntegerField(default=5)
    #marks_obtained = models.IntegerField(default=0)

    #def __str__(self):
     #   return str(self.user)


    #class Meta:
     #   verbose_name = 'Video'
      #  verbose_name_plural = 'Videos'


#class Internship(models.Model):
 #   internship_id = models.IntegerField(primary_key=True)
  #  title = models.CharField(max_length=30)
   # description = models.CharField(max_length=120)
    #image = models.ImageField(blank=True, upload_to="images/")

    #def __str__(self):
     #   return self.title


#class Mentorship(models.Model):
 #   mentorship_id = models.IntegerField(primary_key=True)
  #  title1 = models.CharField(max_length=30)
   # image = models.ImageField(blank=True, upload_to="images/")

    #def __str__(self):
     #   return self.title1


#class Question(models.Model):
 #   qid = models.AutoField(primary_key=True)
  #  question_title = models.CharField(max_length=100)
   # question_text = models.TextField(max_length=50000)
    #date_posted = models.DateTimeField(auto_now_add=True)
    #posted_by = models.TextField(max_length=20)
    #slug = models.SlugField(max_length=40)

    #def save(self, *args, **kwargs):
     #   self.slug = slugify(self.question_title)
      #  super(Question, self).save(*args, **kwargs)

#class Answer(models.Model):
 #   aid = models.AutoField(primary_key=True)
  #  qid = models.ForeignKey(Question, on_delete=models.CASCADE)
   # answer_text = models.TextField(max_length=50000)
    #date_posted = models.DateTimeField(auto_now_add=True)
    #posted_by = models.TextField(max_length=20)





