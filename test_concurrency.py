import asyncio
import time

import httpx


URL = "http://127.0.0.1:8000/query"


async def send_request(client, index):

    start = time.perf_counter()

    response = await client.post(
        URL,
        json={
            "question":
                f"What is the annual leave policy? Request {index}"
        },
        timeout=200
    )

    latency = (
        time.perf_counter() - start
    )

    return {
        "index": index,
        "status": response.status_code,
        "latency": latency
    }


async def main():

    async with httpx.AsyncClient() as client:

        tasks = [
            send_request(client, index)
            for index in range(8)
        ]

        results = await asyncio.gather(
            *tasks
        )

    for result in results:

        print(
            f"Request {result['index']}: "
            f"status={result['status']} "
            f"latency={result['latency']:.2f}s"
        )


if __name__ == "__main__":

    asyncio.run(main())