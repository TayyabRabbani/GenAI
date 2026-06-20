from django.contrib import admin

from .models import SolvedProblem


@admin.register(SolvedProblem)
class SolvedProblemAdmin(admin.ModelAdmin):
    list_display = ("name", "topic", "difficulty", "score", "solved_at")
    list_filter = ("topic", "difficulty")
    search_fields = ("name", "statement")
