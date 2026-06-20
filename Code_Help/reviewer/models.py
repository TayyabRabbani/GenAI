import hashlib

from django.db import models

from .topics import TOPICS


def statement_hash(statement: str) -> str:
    """Stable identity for a problem = sha1 of its whitespace-normalized statement."""
    norm = " ".join((statement or "").split()).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


class SolvedProblem(models.Model):
    DIFFICULTY_CHOICES = [("Easy", "Easy"), ("Medium", "Medium"), ("Hard", "Hard")]

    name = models.CharField(max_length=255)
    statement = models.TextField()
    solution = models.TextField()
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, default="Medium")
    topic = models.CharField(max_length=40, choices=[(t, t) for t in TOPICS], default="Miscellaneous")

    # Review snapshot captured when the problem was solved.
    score = models.FloatField(null=True, blank=True)
    time_complexity = models.CharField(max_length=60, blank=True)
    space_complexity = models.CharField(max_length=60, blank=True)
    summary = models.TextField(blank=True)

    # Identity (dedupe re-solves of the same statement).
    statement_sha = models.CharField(max_length=40, unique=True, db_index=True)

    solved_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-solved_at"]

    def __str__(self):
        return f"{self.name} ({self.topic})"
