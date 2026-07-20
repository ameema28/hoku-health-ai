import datetime
from jose import jwt
from app.core.config import settings


def gen_token(i: int) -> str:
    return jwt.encode(
        {"sub": str(i), "id": i, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def main() -> None:
    for i in (1, 2):
        print("Bearer " + gen_token(i))


if __name__ == "__main__":
    main()
