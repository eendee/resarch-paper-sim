import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';

import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';
import { HomePageComponent } from './component/home-page/home-page.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { ComparePageComponent } from './component/compare-page/compare-page.component';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import {MatToolbarModule} from '@angular/material/toolbar'; 
import {MatButtonModule} from '@angular/material/button'; 
import {MatListModule} from '@angular/material/list'; 
import {MatCardModule} from '@angular/material/card'; 
import {MatBadgeModule} from '@angular/material/badge'; 
import {MatExpansionModule} from '@angular/material/expansion'; 
import {MatProgressBarModule} from '@angular/material/progress-bar'; 

@NgModule({
  declarations: [
    AppComponent,
    HomePageComponent,
    ComparePageComponent
  ],
  imports: [
    MatProgressBarModule,
    MatExpansionModule,
    MatListModule,
    MatBadgeModule,
    MatFormFieldModule,
    MatButtonModule,
    MatCardModule,
    MatToolbarModule,
    MatSelectModule,
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
