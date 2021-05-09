# Generated by Django 3.0.2 on 2020-07-09 06:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reg', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Workshop',
            fields=[
                ('workshop_id', models.IntegerField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=30)),
                ('description', models.CharField(max_length=100)),
                ('link', models.CharField(max_length=50)),
                ('image', models.ImageField(blank=True, upload_to='images/')),
            ],
        ),
    ]