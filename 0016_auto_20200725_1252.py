# Generated by Django 3.0.8 on 2020-07-25 07:22

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('reg', '0015_remove_user_dob'),
    ]

    operations = [
        migrations.CreateModel(
            name='Categories_dept',
            fields=[
                ('dept_id', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('dept_title', models.CharField(max_length=50)),
                ('image', models.ImageField(blank=True, upload_to='images/')),
                ('link', models.CharField(max_length=25)),
            ],
        ),
        migrations.CreateModel(
            name='Newsletter',
            fields=[
                ('newsletter_id', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=100)),
                ('description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Sample_test',
            fields=[
                ('question_id', models.IntegerField(primary_key=True, serialize=False)),
                ('question', models.TextField(max_length=200)),
                ('option_1', models.CharField(max_length=100)),
                ('option_2', models.CharField(max_length=100)),
                ('option_3', models.CharField(max_length=100)),
                ('option_4', models.CharField(max_length=100)),
                ('right_option', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Test_categories',
            fields=[
                ('category_id', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('category_title', models.CharField(max_length=50)),
                ('image', models.ImageField(blank=True, upload_to='images/')),
                ('link', models.CharField(max_length=25)),
            ],
        ),
        migrations.AlterModelOptions(
            name='online_course_video',
            options={},
        ),
        migrations.CreateModel(
            name='Test_series',
            fields=[
                ('test_id', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=50)),
                ('description', models.TextField()),
                ('image', models.ImageField(blank=True, upload_to='images/')),
                ('sample_test_link', models.CharField(max_length=25)),
                ('test_link', models.CharField(max_length=25)),
                ('dept_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.Categories_dept')),
            ],
        ),
        migrations.CreateModel(
            name='Sample_test_result',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('max_marks', models.IntegerField(default=5)),
                ('marks_obtained', models.IntegerField(default=0)),
                ('test_series_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.Test_series')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.User')),
            ],
            options={
                'verbose_name': 'Video',
                'verbose_name_plural': 'Videos',
            },
        ),
        migrations.CreateModel(
            name='Sample_test_response',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('response', models.IntegerField()),
                ('question_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.Sample_test')),
                ('test_series_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.Test_series')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.User')),
            ],
        ),
        migrations.AddField(
            model_name='sample_test',
            name='test_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.Test_series'),
        ),
        migrations.AddField(
            model_name='categories_dept',
            name='category_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='reg.Test_categories'),
        ),
    ]
