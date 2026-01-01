export async function onRequest(context: any) {
  const { request } = context;

  const url = new URL(request.url);

  // remove /api from path
  const backendPath = url.pathname.replace(/^\/api/, "");

  const backendUrl = `http://4.240.107.18${backendPath}`;

  const init: RequestInit = {
    method: request.method,
    headers: request.headers,
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.text();
  }

  const response = await fetch(backendUrl, init);

  return response;
}
