from django.urls import path

from . import views

app_name = "reviewer"

urlpatterns = [
    path("", views.index, name="index"),
    path("review/", views.review, name="review"),
    path("save/", views.save, name="save"),
    path("dashboard/", views.dashboard, name="dashboard"),
]
