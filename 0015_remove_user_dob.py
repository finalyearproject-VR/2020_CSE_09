# Generated by Django 3.0.6 on 2020-07-20 14:13

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('reg', '0014_mentorship'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='user',
            name='dob',
        ),
    ]