import { io, Socket } from "socket.io-client";

let socket: Socket;

export const initSocket = () => {
  socket = io("http://127.0.0.1:4000", {
    transports: ["websocket"],
    autoConnect: true,
  });

  socket.on("connect", () => {
    console.log("Connected to server");
  });

  socket.on("disconnect", () => {
    console.log("Disconnected from server");
  });

  return socket;
};

export const getSocket = () => {
  if (!socket) {
    return initSocket();
  }
  return socket;
};
