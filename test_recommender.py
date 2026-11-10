# Test recommendation backend

from recommender import recommend_hybrid

recommendations = recommend_hybrid(
    user_id=196,
    top_n=10
)

print("\nPersonalized Movie Recommendations:")
print(recommendations.to_string(index=False))