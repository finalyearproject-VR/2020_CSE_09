# Generated by Django 3.1.7 on 2021-04-23 09:13

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('reg', '0019_newsletter_videos1'),
    ]

    operations = [
        migrations.CreateModel(
            name='Graduation_vedio_videos1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('usn', models.CharField(max_length=300)),
                ('videos1', models.FileField(upload_to='videos1/')),
            ],
        ),
        migrations.RemoveField(
            model_name='answer',
            name='qid',
        ),
        migrations.RemoveField(
            model_name='categories_dept',
            name='category_id',
        ),
        migrations.RemoveField(
            model_name='enrolled',
            name='email',
        ),
        migrations.DeleteModel(
            name='Institutes',
        ),
        migrations.DeleteModel(
            name='Internship',
        ),
        migrations.DeleteModel(
            name='Mentorship',
        ),
        migrations.RemoveField(
            model_name='newsletter_videos1',
            name='newsletter_id',
        ),
        migrations.RemoveField(
            model_name='offline_description',
            name='off_id',
        ),
        migrations.RemoveField(
            model_name='sample_test',
            name='test_id',
        ),
        migrations.RemoveField(
            model_name='sample_test_response',
            name='question_id',
        ),
        migrations.RemoveField(
            model_name='sample_test_response',
            name='test_series_id',
        ),
        migrations.RemoveField(
            model_name='sample_test_response',
            name='user',
        ),
        migrations.RemoveField(
            model_name='sample_test_result',
            name='test_series_id',
        ),
        migrations.RemoveField(
            model_name='sample_test_result',
            name='user',
        ),
        migrations.RemoveField(
            model_name='test_series',
            name='dept_id',
        ),
        migrations.DeleteModel(
            name='Workshop',
        ),
        migrations.RenameField(
            model_name='graduation_vedio',
            old_name='description',
            new_name='college',
        ),
        migrations.RenameField(
            model_name='graduation_vedio',
            old_name='newsletter_id',
            new_name='graduationstudent_id',
        ),
        migrations.RenameField(
            model_name='graduation_vedio',
            old_name='title',
            new_name='usn',
        ),
        migrations.RenameModel(
            old_name='Newsletter',
            new_name='Graduation_vedio',
        ),
        migrations.DeleteModel(
            name='Answer',
        ),
        migrations.DeleteModel(
            name='Categories_dept',
        ),
        migrations.DeleteModel(
            name='Enrolled',
        ),
        migrations.DeleteModel(
            name='Newsletter_videos1',
        ),
        migrations.DeleteModel(
            name='Offline_course',
        ),
        migrations.DeleteModel(
            name='Offline_description',
        ),
        migrations.DeleteModel(
            name='Question',
        ),
        migrations.DeleteModel(
            name='Sample_test',
        ),
        migrations.DeleteModel(
            name='Sample_test_response',
        ),
        migrations.DeleteModel(
            name='Sample_test_result',
        ),
        migrations.DeleteModel(
            name='Test_categories',
        ),
        migrations.DeleteModel(
            name='Test_series',
        ),
        migrations.AddField(
            model_name='graduation_vedio_videos1',
            name='graduationstudent_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.graduation_vedio'),
        ),
    ]
