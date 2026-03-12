let menuIcon= document.querySelector('#menu-icon');
let nav= document.querySelector('.nav');

let navCenter = document.querySelector('.nav-center');


menuIcon.onclick = () => {
    menuIcon.classList.toggle('bx-x');
    nav.classList.toggle('active');

    navCenter.classList.toggle('active');
}