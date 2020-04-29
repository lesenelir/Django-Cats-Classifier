# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Catinfo',
            fields=[
                ('id', models.AutoField(verbose_name='ID', primary_key=True, serialize=False, auto_created=True)),
                ('name', models.CharField(max_length=10)),
                ('nameinfo', models.CharField(max_length=1000)),
                ('feature', models.CharField(max_length=1000)),
                ('livemethod', models.CharField(max_length=1000)),
                ('feednn', models.CharField(max_length=1000)),
                ('feedmethod', models.CharField(max_length=1000)),
            ],
        ),
    ]
